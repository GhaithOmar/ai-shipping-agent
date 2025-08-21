#!/usr/bin/env python3
# scripts/data_prep/build_sft.py
# Build an SFT dataset (mini or full) from Kaggle + HF Bitext with filtering, PII scrub,
# aliasing, de-twitterization, dedup, and rebalancing.

import os, re, json, random, argparse, unicodedata
from pathlib import Path
from typing import List, Dict
import pandas as pd
from datasets import load_dataset
import yaml

random.seed(42)

# --- Regex patterns ---
PII_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PII_PHONE = re.compile(r'(?:(?:\+?\d{1,3})?[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}')
ORDER_LIKE = re.compile(r'\b(#?\d{6,}[-]?\d*)\b')
URL = re.compile(r'https?://\S+')
HANDLE = re.compile(r'@\w+')
CARET_TAG = re.compile(r'\^[A-Z]{1,3}\b')

def normalize_text(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def scrub_pii(s: str) -> str:
    s = PII_EMAIL.sub("[REDACTED_EMAIL]", s)
    s = PII_PHONE.sub("[REDACTED_PHONE]", s)
    s = ORDER_LIKE.sub("[REDACTED_ORDER]", s)
    s = URL.sub("", s)
    s = HANDLE.sub("", s)
    s = CARET_TAG.sub("", s)
    return s

def alias_carriers(s: str, names: List[str], alias: str) -> str:
    out = s
    for n in names:
        out = re.sub(rf'\b{re.escape(n)}\b', alias, out, flags=re.IGNORECASE)
    out = re.sub(r'shipping_a', 'Shipping_A', out, flags=re.IGNORECASE)
    return out

def passes_shipping_filter(text: str, keywords: List[str]) -> bool:
    text_low = text.lower()
    return any(k.lower() in text_low for k in keywords)

def detwitterize_agent(text: str, prepend_steps: bool) -> str:
    t = URL.sub("", text)
    t = HANDLE.sub("", t)
    t = CARET_TAG.sub("", t)
    t = re.sub(r'#\w+', '', t)
    t = re.sub(r'\bDM\b.*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\b[dD]irect message\b.*$', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    t = t.replace("pls", "please").replace("thx", "thanks")
    if prepend_steps and "Here's what I'll do:" not in t and "Here’s what I’ll do:" not in t:
        t = ("Thanks for reaching out. Here's what I'll do:\n"
             "1) Check the latest scan and status.\n"
             "2) Confirm delivery address.\n"
             "3) Share an ETA or next step.\n\n") + t
    return t.strip()

def enforce_max_tokens(text: str, max_tokens: int = 1024) -> str:
    max_chars = max_tokens * 4
    return text[:max_chars] if len(text) > max_chars else text

def map_pair(user: str, agent: str, cfg: dict) -> Dict[str, str]:
    user = scrub_pii(normalize_text(user))
    agent = scrub_pii(normalize_text(agent))
    alias_list = cfg["anonymize"]["carrier_aliases"]
    alias_to = cfg["anonymize"]["replace_with"]
    user  = alias_carriers(user,  alias_list, alias_to)
    agent = alias_carriers(agent, alias_list, alias_to)
    if cfg["tone"]["normalize_agent_style"]:
        agent = detwitterize_agent(agent, prepend_steps=cfg["tone"]["prepend_steps"])
    user  = enforce_max_tokens(user,  cfg["output"]["max_tokens"])
    agent = enforce_max_tokens(agent, cfg["output"]["max_tokens"])
    return {"input": user, "assistant_response": agent}

# --- Kaggle loader (dtype-safe) ---
def load_kaggle(path: str, mapping: dict) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={mapping["tweet_id_col"]: "string", mapping["response_tweet_id_col"]: "string"},
        keep_default_na=True
    )
    user_col, inbound_col, tweet_id_col, resp_id_col = (
        mapping["user_col"], mapping["inbound_col"], mapping["tweet_id_col"], mapping["response_tweet_id_col"]
    )
    def norm_bool_col(s): return (
        s.astype(str).str.strip().str.lower()
         .map({"true": True, "false": False, "1": True, "0": False})
         .fillna(False).astype(bool)
    )
    def to_str_id(series): 
        out = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        return out.mask(out.isin(["nan","NaN","None"]), "")
    df[inbound_col]  = norm_bool_col(df[inbound_col])
    df[tweet_id_col] = to_str_id(df[tweet_id_col])
    df[resp_id_col]  = to_str_id(df[resp_id_col])
    customers = df[(df[inbound_col]==True) & (df[resp_id_col].ne(""))][[tweet_id_col, resp_id_col, user_col]].rename(columns={user_col:"input_raw"})
    companies = df[(df[inbound_col]==False)][[tweet_id_col, user_col]].rename(columns={tweet_id_col:"reply_tweet_id", user_col:"assistant_raw"})
    companies["reply_tweet_id"] = to_str_id(companies["reply_tweet_id"])
    pairs = customers.merge(companies, left_on=resp_id_col, right_on="reply_tweet_id", how="left", validate="many_to_one").dropna(subset=["assistant_raw"])
    return pairs[["input_raw","assistant_raw"]]

def load_hf(repo_id: str, split: str, mapping: dict) -> pd.DataFrame:
    ds = load_dataset(repo_id, split=split)
    return ds.to_pandas()[[mapping["user_col"], mapping["agent_col"]]].rename(columns={mapping["user_col"]:"input_raw", mapping["agent_col"]:"assistant_raw"})

# --- Dedup ---
def _norm(s: str) -> str:
    s = s.casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()
def dedup_pairs(pairs: List[Dict[str,str]]) -> List[Dict[str,str]]:
    seen, out = set(), []
    for ex in pairs:
        key = (_norm(ex["input"]), _norm(ex["assistant_response"]))
        if key in seen: continue
        seen.add(key); out.append(ex)
    return out

# --- Main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg["output"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    mini_size = int(cfg["output"]["mini_size"]); val_ratio = float(cfg["output"]["val_ratio"])
    keywords = cfg["filters"]["shipping_keywords"]

    all_examples = []
    for ds_cfg in cfg["datasets"]:
        if ds_cfg["kind"].startswith("kaggle"):
            df = load_kaggle(ds_cfg["path"], ds_cfg["mapping"])
        elif ds_cfg["kind"]=="hf":
            df = load_hf(ds_cfg["hf_repo"], ds_cfg["split"], ds_cfg["mapping"])
        else:
            continue
        # filter
        mask = df["input_raw"].astype(str).apply(lambda s: passes_shipping_filter(s, keywords)) | \
               df["assistant_raw"].astype(str).apply(lambda s: passes_shipping_filter(s, keywords))
        df = df[mask]
        for _, row in df.iterrows():
            all_examples.append(map_pair(str(row["input_raw"]), str(row["assistant_raw"]), cfg))

    # dedup
    all_examples = dedup_pairs(all_examples)

    # --- Rebalancing ---
    ratios = cfg.get("ratios", {"bitext_hf":0.7, "kaggle_twitter":0.2, "synthetic":0.1})
    buckets = {k: [] for k in ratios}
    for ex in all_examples:
        src = "bitext_hf" if "bitext" in ex["assistant_response"].lower() else "kaggle_twitter"
        buckets[src].append(ex)
    final = []
    for src, frac in ratios.items():
        n = int(mini_size * frac)
        random.shuffle(buckets.get(src, []))
        final.extend(buckets.get(src, [])[:n])
    if len(final) < mini_size:  # top up
        random.shuffle(all_examples)
        final.extend(all_examples[:mini_size - len(final)])
    random.shuffle(final)
    examples = final[:mini_size]

    n_val = max(1, int(len(examples)*val_ratio))
    val, train = examples[:n_val], examples[n_val:]

    with open(out_dir/"mini_sft.jsonl","w",encoding="utf-8") as f:
        for ex in train: f.write(json.dumps(ex,ensure_ascii=False)+"\n")
    with open(out_dir/"mini_sft_val.jsonl","w",encoding="utf-8") as f:
        for ex in val: f.write(json.dumps(ex,ensure_ascii=False)+"\n")
    manifest = {"total_pairs":len(examples),"train_pairs":len(train),"val_pairs":len(val),"ratios":ratios}
    with open(out_dir/"manifest.json","w",encoding="utf-8") as f:
        json.dump(manifest,f,indent=2,ensure_ascii=False)
    print(f"Wrote {len(train)} train and {len(val)} val examples to {out_dir}")

if __name__=="__main__":
    main()
