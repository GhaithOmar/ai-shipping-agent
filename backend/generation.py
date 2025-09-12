import re
import threading
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
)

SYSTEM_PREFIX = (
    "You are a shipping support assistant. Always ask for missing IDs, never include links, "
    "never claim live tracking. Keep answers concise with 2–4 bullet steps and defer facts to retrieval."
)

# Caches
_TOK: PreTrainedTokenizerBase | None = None
_MODEL: nn.Module | None = None
_GEN_CFG: GenerationConfig | None = None
_PROCESSORS: LogitsProcessorList | None = None

# ---------------------- helpers ----------------------
def _bf16_supported() -> bool:
    try:
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        return False

_BAD_PATTERNS = [
    r"(?i)live\s+tracking",
    r"(?i)click\s+here",
]

def _has_bad_pattern(text: str) -> bool:
    return any(re.search(p, text) for p in _BAD_PATTERNS)

def _postprocess(text: str) -> str:
    # minimal guardrails
    if _has_bad_pattern(text):
        text = re.sub(r"(?i)click\s+here.*", "", text).strip()
    # simple bullet cleanup
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets: List[str] = []
    for ln in lines:
        if not ln.startswith(("-", "*")):
            ln = f"- {ln}"
        bullets.append(ln)
    # link refusal up top
    if not any(re.search(r"can('?|no)t share links|cannot share links", b, re.I) for b in bullets):
        bullets.insert(0, "- I can’t share tracking links. Use the carrier’s official site/app with your waybill number.")
    # ask for IDs if missing
    if not any(re.search(r"(tracking|waybill|order)\s*(number|id)", b, re.I) for b in bullets):
        bullets.insert(0, "- Please share your tracking/waybill number and the carrier (e.g., Shipping_A).")
    return "\n".join(bullets[:4])

def _generate_in_thread(m: nn.Module, kwargs: Dict[str, Any]) -> None:
    m.generate(**kwargs)

# ---------------------- load ----------------------
def load_model_and_tokenizer(base_id: str, adapter: Optional[str], hf_token: Optional[str]) -> Tuple[PreTrainedTokenizerBase, nn.Module]:
    global _TOK, _MODEL, _GEN_CFG, _PROCESSORS
    if _TOK is not None and _MODEL is not None and _GEN_CFG is not None and _PROCESSORS is not None:
        return _TOK, _MODEL

    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True, token=hf_token)
    tok.pad_token = tok.eos_token

    torch_dtype = torch.bfloat16 if _bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    ).eval()

    if adapter:
        model: nn.Module = PeftModel.from_pretrained(base, adapter, token=hf_token).eval()
    else:
        model = base

    tok_any: Any = tok
    bad_enc = tok_any.batch_encode_plus(  # type: ignore[operator]
        ["http", "https", "://", "www."],
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    bad_words_ids = bad_enc["input_ids"]

    processors = LogitsProcessorList([NoBadWordsLogitsProcessor(bad_words_ids=bad_words_ids, eos_token_id=tok.eos_token_id)])
    gen_cfg = GenerationConfig(
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.05,
        no_repeat_ngram_size=4,
    )

    _TOK, _MODEL, _GEN_CFG, _PROCESSORS = tok, model, gen_cfg, processors
    return tok, model

# ---------------------- public APIs ----------------------
def infer_guarded(
    user_msg: str,
    top_k_context: Optional[List[str]],
    tok: PreTrainedTokenizerBase,
    model: nn.Module,
    provided_tracking: Optional[str] = None,
) -> str:
    messages = [{"role": "system", "content": SYSTEM_PREFIX}]
    if top_k_context:
        ctx = "(Context — citations):\n" + "\n".join(f"- {c}" for c in top_k_context)
        messages.append({"role": "user", "content": ctx})
    messages.append({"role": "user", "content": user_msg})

    prompt = tok.apply_chat_template(  # type: ignore[attr-defined]
        messages, tokenize=False, add_generation_prompt=True
    )

    tok_any: Any = tok
    enc = tok_any.encode_plus(prompt, return_tensors="pt")  # type: ignore[operator]

    inputs = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}


    assert _GEN_CFG is not None and _PROCESSORS is not None
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            generation_config=_GEN_CFG,
            logits_processor=_PROCESSORS,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_tokens = out[0, prompt_len:]
    raw = tok.decode(gen_tokens, skip_special_tokens=True).strip()
    return _postprocess(raw)

def stream_guarded(user_msg: str, top_k_context: List[str], tracking_id: Optional[str] = None):
    assert _TOK is not None and _MODEL is not None
    tok = cast(PreTrainedTokenizerBase, _TOK)
    model = cast(nn.Module, _MODEL)

    messages = [{"role": "system", "content": SYSTEM_PREFIX}]
    if top_k_context:
        ctx = "(Context — citations):\n" + "\n".join(f"- {c}" for c in top_k_context)
        messages.append({"role": "user", "content": ctx})
    messages.append({"role": "user", "content": user_msg})

    prompt = tok.apply_chat_template(  # type: ignore[attr-defined]
        messages, tokenize=False, add_generation_prompt=True
    )
    tok_any: Any = tok
    enc = tok_any.encode_plus(prompt, return_tensors="pt")  # type: ignore[operator]

    inputs = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}


    streamer = TextIteratorStreamer(cast(AutoTokenizer, tok), skip_prompt=True, skip_special_tokens=True)

    gen_kwargs: Dict[str, Any] = dict(
        **inputs,
        max_new_tokens=384,
        do_sample=False,
        streamer=streamer,
        repetition_penalty=1.05,
        no_repeat_ngram_size=4,
    )

    t = threading.Thread(target=_generate_in_thread, args=(model, gen_kwargs))
    t.start()

    for piece in streamer:
        yield piece
