"""Large test-set evaluation for the research article.

Samples up to N questions per (task, source) group from the test split with a fixed seed, so every model is
scored on the same items, answers them with greedy decoding (batched, on the GPU), and writes one JSON record
per question: prompt, reference, model answer, task, source, answerable.

usage (from the repo root): python scripts/evaluate_full.py <hf_model_dir> <out.jsonl> [N=200]
"""
import json
import random
import sys
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, out_path = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
TEST = "data/energydata_sml_large_test.jsonl"
BATCH = 16

groups = defaultdict(list)
for line in open(TEST):
    r = json.loads(line)
    groups[(r["task"], r.get("source") or "uci")].append(r)
rng = random.Random(2026)
items = []
for key in sorted(groups):
    rows = groups[key]
    items += rng.sample(rows, min(N, len(rows)))
print(f"{len(items)} questions from {len(groups)} groups", flush=True)

tok = AutoTokenizer.from_pretrained(model_dir)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).cuda().eval()
eos = tok.convert_tokens_to_ids("<|im_end|>")

texts = [f"<|im_start|>user\n{r['prompt']}<|im_end|>\n<|im_start|>assistant\n" for r in items]
order = sorted(range(len(items)), key=lambda i: len(texts[i]))
answers = [None] * len(items)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=170, do_sample=False, eos_token_id=eos,
                             pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            answers[i] = tok.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        if (b // BATCH) % 25 == 0:
            print(f"{b + len(idx)}/{len(items)}", flush=True)

with open(out_path, "w") as f:
    for r, a in zip(items, answers):
        f.write(json.dumps({"task": r["task"], "source": r.get("source") or "uci", "answerable": r["answerable"],
                            "prompt": r["prompt"], "reference": r["completion"].strip(), "model": a},
                           ensure_ascii=False) + "\n")
print("wrote", out_path)
