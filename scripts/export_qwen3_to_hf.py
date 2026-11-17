"""Convert the from-scratch checkpoint (qwen3_slm.pt) into a standard Hugging Face Qwen3ForCausalLM folder.

The notebook's Qwen3Model matches the Hugging Face Qwen3 layout (GQA, per-head q/k RMSNorm, SwiGLU, rotate-half
RoPE) with one difference: it multiplies token embeddings by sqrt(d_model) before the first block. Hugging Face
Qwen3 does not, so the scale is folded into the input embedding table and the output head keeps the unscaled
table (tie_word_embeddings = False). The result loads with AutoModelForCausalLM and converts to GGUF for Ollama.

usage (from the repo root): python scripts/export_qwen3_to_hf.py [checkpoint.pt] [config.json] [out_dir]
"""
import json
import math
import os
import sys

import torch
from transformers import AutoTokenizer

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "out/qwen3_scratch/qwen3_slm.pt"
cfg_path = sys.argv[2] if len(sys.argv) > 2 else "out/qwen3_scratch/hf_export/config.json"
out_dir = sys.argv[3] if len(sys.argv) > 3 else "out/qwen3_scratch/hf_qwen3"

cfg = json.load(open(cfg_path))
mc = cfg["model_config"]
d, n_heads, n_kv = mc["d_model"], mc["n_heads"], mc["n_kv_groups"]
head_dim = d // n_heads

sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
sd = {k.removeprefix("_orig_mod."): v.float() for k, v in sd.items()}

emb = sd["tok_emb.weight"]
new = {
    "model.embed_tokens.weight": emb * math.sqrt(d),
    "lm_head.weight": emb.clone(),
    "model.norm.weight": sd["final_norm.weight"],
}
names = {
    "attention.W_query": "self_attn.q_proj",
    "attention.W_key": "self_attn.k_proj",
    "attention.W_value": "self_attn.v_proj",
    "attention.out_proj": "self_attn.o_proj",
    "attention.q_norm": "self_attn.q_norm",
    "attention.k_norm": "self_attn.k_norm",
    "feed_forward.gate_proj": "mlp.gate_proj",
    "feed_forward.up_proj": "mlp.up_proj",
    "feed_forward.down_proj": "mlp.down_proj",
    "norm1": "input_layernorm",
    "norm2": "post_attention_layernorm",
}
for i in range(mc["n_layers"]):
    for old, hf in names.items():
        new[f"model.layers.{i}.{hf}.weight"] = sd[f"blocks.{i}.{old}.weight"]

used = {"tok_emb.weight", "final_norm.weight", "out_head.weight"} | {
    f"blocks.{i}.{old}.weight" for i in range(mc["n_layers"]) for old in names
}
leftover = set(sd) - used
assert not leftover, f"unmapped weights: {sorted(leftover)}"

tok = AutoTokenizer.from_pretrained(os.path.dirname(cfg_path))
hf_config = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "vocab_size": mc["vocab_size"],
    "hidden_size": d,
    "intermediate_size": mc["d_ff"],
    "num_hidden_layers": mc["n_layers"],
    "num_attention_heads": n_heads,
    "num_key_value_heads": n_kv,
    "head_dim": head_dim,
    "hidden_act": "silu",
    "max_position_embeddings": mc["max_seq_len"],
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_sliding_window": False,
    "sliding_window": None,
    "max_window_layers": mc["n_layers"],
    "initializer_range": 0.02,
    "torch_dtype": "float32",
    "bos_token_id": None,
    "eos_token_id": tok.convert_tokens_to_ids("<|im_end|>"),
    "pad_token_id": tok.convert_tokens_to_ids("<|endoftext|>"),
}

from transformers import Qwen3Config, Qwen3ForCausalLM

model = Qwen3ForCausalLM(Qwen3Config(**hf_config))
missing, unexpected = model.load_state_dict(new, strict=False)
missing = [m for m in missing if "rotary_emb" not in m]
assert not missing and not unexpected, (missing, unexpected)

# The training prompt format has no system turn and no <think> block, so give the export a matching chat template.
tok.chat_template = (
    "{% for m in messages %}{% if m['role'] == 'user' %}<|im_start|>user\n{{ m['content'] }}<|im_end|>\n"
    "{% elif m['role'] == 'assistant' %}<|im_start|>assistant\n{{ m['content'] }}<|im_end|>\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
os.makedirs(out_dir, exist_ok=True)
model.save_pretrained(out_dir, safe_serialization=True)
model.generation_config.eos_token_id = hf_config["eos_token_id"]
model.generation_config.pad_token_id = hf_config["pad_token_id"]
model.generation_config.do_sample = False
model.generation_config.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print("wrote", out_dir, "-", sum(p.numel() for p in model.parameters()) / 1e6, "M parameters (untied)")
