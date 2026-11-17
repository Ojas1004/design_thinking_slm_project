"""Merge the LoRA adapter (Qwen2.5-0.5B-Instruct + energy LoRA) into a plain Hugging Face model for GGUF / Ollama.

usage (from the repo root, after train.py): python scripts/merge_lora.py [adapter_dir] [out_dir]
"""
import os
import sys

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

adapter = sys.argv[1] if len(sys.argv) > 1 else "out/qwen0.5b-energy-lora/final"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "out/lora_merged"

tok = AutoTokenizer.from_pretrained(adapter)
model = AutoPeftModelForCausalLM.from_pretrained(adapter, dtype=torch.float32)
merged = model.merge_and_unload()
os.makedirs(out_dir, exist_ok=True)
merged.save_pretrained(out_dir, safe_serialization=True)
tok.save_pretrained(out_dir)

# sanity check: the merged model answers like the adapter model
q = "Appliances: 250 Wh at 2016-05-18 19:10. Is that unusually high?"
ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                              return_tensors="pt", return_dict=True)
out = merged.generate(**ids, max_new_tokens=120, do_sample=False)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
print("wrote", out_dir)
