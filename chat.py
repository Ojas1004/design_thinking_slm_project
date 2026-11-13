"""Ask the trained smart-home energy model questions in the terminal.

Usage: python chat.py   (empty line to quit)
Loads the local adapter from out/qwen0.5b-energy-lora/final if present,
otherwise downloads it from the Hugging Face Hub.
"""
import os

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

local = "out/qwen0.5b-energy-lora/final"
path = local if os.path.isdir(local) else "Ojas1004/smart-home-energy-slm"

tok = AutoTokenizer.from_pretrained(path)
model = AutoPeftModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto").eval()
print(f"Loaded {path}. Type a question (empty line to quit).")

while True:
    try:
        q = input("\nYou: ").strip()
    except EOFError:
        break
    if not q:
        break
    ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    out = model.generate(**ids, max_new_tokens=200, do_sample=False)
    print("Model:", tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
