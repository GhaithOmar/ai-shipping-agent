# backend/generation.py
import os
import re
from typing import List, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    GenerationConfig, LogitsProcessorList, NoBadWordsLogitsProcessor
)
from peft import PeftModel
# --- streaming generation (legacy path) ---
from transformers import TextIteratorStreamer
import threading

# =========================
# System policy / prompt
# =========================
SYSTEM_PREFIX = (
    "You are a shipping support assistant. "
    "Ask for missing IDs, never include links, never claim live tracking. "
    "Keep answers concise with 2–4 bullet steps and defer facts to retrieval."
)

# Token-level blocks for links/handles/marketing phrasing
BAD_PATTERNS = [
    "http://", "https://", "www.", ".com", ".net", ".org", ".io", ".co", ".ly",
    "@", " DM ", "direct message", "^", " #", " link ", " url ", " website "
]

# Singleton cache
_TOK = None
_MODEL = None


def _bf16_supported() -> bool:
    try:
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        return False


def load_model_and_tokenizer(
    base_model_id: str,
    adapter_id_or_path: str,
    device_map: str = "auto"
) -> Tuple[AutoTokenizer, PeftModel]:
    """
    Loads base model (4-bit if CUDA available) + PEFT adapter, returns (tokenizer, peft_model).
    Reuses a process-wide singleton so repeated imports don't reload weights.
    """
    global _TOK, _MODEL
    if _TOK is not None and _MODEL is not None:
        return _TOK, _MODEL

    is_cuda = torch.cuda.is_available()

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    tok.pad_token = tok.eos_token

    # Base model
    if is_cuda:
        # 4-bit quant for GPU
        from transformers import BitsAndBytesConfig
        compute_dtype = torch.bfloat16 if _bf16_supported() else torch.float16
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        # CPU fallback (slow; for dev/test only)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    # Attach LoRA adapter (HF repo ID or local path)
    model = PeftModel.from_pretrained(base, adapter_id_or_path).eval()

    # Build logits processors once
    bad_words_ids = [ids for pat in BAD_PATTERNS if (ids := tok.encode(pat, add_special_tokens=False))]
    processors = LogitsProcessorList([NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id=tok.eos_token_id)])

    # Deterministic decoding config (silences temp/top_p warnings)
    gen_cfg = GenerationConfig(
        do_sample=False,
        no_repeat_ngram_size=4,
        repetition_penalty=1.15,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        max_new_tokens=140,
    )

    # Stash for reuse
    model._gen_cfg = gen_cfg
    model._processors = processors

    _TOK, _MODEL = tok, model
    return tok, model


# =========================
# Intent detection
# =========================
TRACK_INTENT = re.compile(
    r"\b(track(ing)?|status|where\s+(is|’s|'s)|locate|find|scan(s)?|last\s+(two|2)\s+scans?|update|trace|follow\s*up)\b",
    re.I
)
LINK_INTENT = re.compile(r"\b(link|url|website|web\s*site)\b", re.I)
DELIVERY_ISSUE_INTENT = re.compile(
    r"\b(delivered|not\s+received|missing|lost|stolen|proof\s+of\s+delivery|pod|misdeliver(ed)?|left\s+at)\b",
    re.I
)
ADDRESS_CHANGE_INTENT = re.compile(r"\b(address|postcode|zip|postal\s+code|suite|apt|apartment|unit)\b", re.I)
CUSTOMS_INTENT = re.compile(r"\b(customs|duty|tariff|hs\s*code|clearance|invoice|id|proof\s+of\s+payment|declaration)\b", re.I)


def infer_intent(user_text: str) -> str:
    t = user_text.lower()
    if LINK_INTENT.search(t): return "link_request"
    if TRACK_INTENT.search(t): return "tracking"
    if DELIVERY_ISSUE_INTENT.search(t): return "delivery_issue"
    if ADDRESS_CHANGE_INTENT.search(t): return "address"
    if CUSTOMS_INTENT.search(t): return "customs"
    return "general"


def has_any_id(text: str, provided_tracking: Optional[str]) -> bool:
    if provided_tracking and len(provided_tracking.strip()) >= 8:
        return True
    t = text.strip()
    # AWB/Waybill/Tracking/Order patterns (alnum with 8+ chars)
    if re.search(r"(?i)\b(awb|waybill|tracking|order)\s*(no\.?|number|id)?\s*[:#-]?\s*[A-Z0-9\-]{8,}", t):
        return True
    # bare long numbers (8+)
    if re.search(r"\b\d{8,}\b", t):
        return True
    return False


def _missing_id(user_text: str, provided_tracking: Optional[str] = None) -> bool:
    """Intent-aware check: when do we require a tracking/waybill ID?"""
    intent = infer_intent(user_text)
    has_id = has_any_id(user_text, provided_tracking)

    # Require ID for these intents
    if intent in {"tracking", "link_request", "delivery_issue", "address"}:
        return not has_id

    # For customs/general info, don't force ID unless they actually ask to check status
    return False


def _wants_link(user_text: str) -> bool:
    return bool(LINK_INTENT.search(user_text))


# =========================
# Post-processing
# =========================
def _postprocess(text: str, require_id: bool, refuse_link: bool) -> str:
    # Strip meta + urls/handles/stock phrases
    text = re.sub(r"(?i)cutting knowledge date:.*|today date:.*", "", text)
    text = re.sub(r"https?://\S+|www\.\S+|\S+\.(com|net|org|io|co|ly)\b", "", text)
    text = re.sub(r"(?i)@\w+|dm|direct message", "", text)
    text = re.sub(r"(?i)(thanks for reaching out|we'?re here to help|view it here).*", "", text)
    text = re.sub(r"[\*\~_]{1,3}", "", text)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Normalize to "- " bullets
    bullets = []
    for ln in lines:
        m = re.match(r"^\s*\d+[.)]\s*(.+)$", ln)  # "1) foo" / "2. bar" -> "- foo"
        if m:
            ln = "- " + m.group(1).strip()
        if re.match(r"^(\-|\*)\s", ln):
            ln = re.sub(r"^\*\s", "- ", ln)
            ln = re.sub(r"\s*[\.·•-]+$", "", ln)
            bullets.append(ln)

    if not bullets:
        # Fallback: make bullets from sentences
        sents = re.split(r"(?<=[.!?])\s+", " ".join(lines))
        bullets = [f"- {s.strip()}" for s in sents if s.strip()][:4]

    # Insert “ask for ID”
    if require_id and not any(re.search(r"(tracking|waybill|order)\s*(number|id)", b.lower()) for b in bullets):
        bullets.insert(0, "- Please share your tracking/waybill number and the carrier (e.g., Shipping_A).")

    # Insert explicit link refusal
    if refuse_link and not any(re.search(r"can('?|no)t share links|cannot share links", b, re.I) for b in bullets):
        bullets.insert(0, "- I can’t share tracking links. Use the carrier’s official site/app with your waybill number.")

    bullets = [b if b.startswith("- ") else "- " + b.lstrip("-* ").strip() for b in bullets[:4]]
    return "\n".join(bullets)


# =========================
# Public inference API
# =========================
def infer_guarded(
    user_msg: str,
    top_k_context: Optional[List[str]],
    tok: AutoTokenizer,
    model: PeftModel,
    provided_tracking: Optional[str] = None
) -> str:
    messages = [{"role": "system", "content": SYSTEM_PREFIX}]
    if top_k_context:
        ctx = "(Context — citations):\n" + "\n".join(f"- {c}" for c in top_k_context)
        messages.append({"role": "user", "content": ctx})
    messages.append({"role": "user", "content": user_msg})

    # Build prompt as STRING, then tokenize to dict
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            generation_config=model._gen_cfg,
            logits_processor=model._processors,
        )

    # Decode ONLY the new tokens after the prompt
    prompt_len = inputs["input_ids"].shape[1]
    gen_tokens = out[0, prompt_len:]
    raw = tok.decode(gen_tokens, skip_special_tokens=True).strip()

    return _postprocess(
        raw,
        require_id=_missing_id(user_msg, provided_tracking),
        refuse_link=_wants_link(user_msg),
    )

def stream_guarded(user_msg: str, top_k_context: list[str], tracking_id: str | None):
    """
    Yields decoded strings as the model generates them.
    Mirrors infer_guarded() policies: system prefix + link suppression.
    """
    global _TOK, _MODEL
    tok = _TOK
    model = _MODEL
    if tok is None or model is None:
        raise RuntimeError("Model not loaded")

    # Build prompt (same structure you use in infer_guarded)
    sys = SYSTEM_PREFIX
    ctx = "\n\n".join([c for c in top_k_context if c]) if top_k_context else ""
    trk = f"\n\n[ParsedTrackingID]: {tracking_id}" if tracking_id else ""
    prompt = f"{sys}\n\n[Context]\n{ctx}{trk}\n\n[User]\n{user_msg}\n\n[Assistant]\n"

    inputs = tok(prompt, return_tensors="pt")
    if model.device.type == "cuda":
        for k in inputs:
            inputs[k] = inputs[k].to(model.device)

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=384,
        do_sample=False,                 # deterministic
        streamer=streamer,
        repetition_penalty=1.05,
        no_repeat_ngram_size=4,
    )

    # Background thread to run generate()
    t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    t.start()

    for piece in streamer:
        yield piece