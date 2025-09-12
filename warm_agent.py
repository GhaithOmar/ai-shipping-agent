import os

from typing import Any 
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hugging Face token (if set in .env)
token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

base = os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
adpt = os.getenv("ADAPTER_ID", "GhaithOmar/ai-shipping-agent-llama3.1-8b-lora-day4")

print("Loading base:", base)
tok = AutoTokenizer.from_pretrained(base, token=token)
tok.pad_token = tok.eos_token

model: Any = AutoModelForCausalLM.from_pretrained(
    base, device_map="cpu", torch_dtype="auto",
    low_cpu_mem_usage=True, trust_remote_code=True, token=token
)

if adpt:
    print("Loading adapter:", adpt)
    model = PeftModel.from_pretrained(model, adpt, token=token)

print("Warmed:", base, "+", adpt)
