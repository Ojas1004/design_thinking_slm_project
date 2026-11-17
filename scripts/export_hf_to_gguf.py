"""Write a Hugging Face Qwen3 or Qwen2 folder as a GGUF file for Ollama / llama.cpp.

Qwen3: the from-scratch model converted by export_qwen3_to_hf.py. Qwen2: the LoRA model merged by merge_lora.py
(attention biases, no q/k norms, output layer tied to the embeddings).

A small, model-specific stand-in for llama.cpp's convert_hf_to_gguf.py: same metadata keys, tensor names and
byte-level BPE vocabulary layout, using the `gguf` package (pip install gguf). Weight matrices are stored as
float16 and norm vectors as float32.

usage (from the repo root): python scripts/export_hf_to_gguf.py [hf_dir] [out.gguf] [model name]
"""
import json
import os
import sys

import gguf
import numpy as np
from safetensors.numpy import load_file

hf_dir = sys.argv[1] if len(sys.argv) > 1 else "out/qwen3_scratch/hf_qwen3"
out_path = sys.argv[2] if len(sys.argv) > 2 else "out/qwen3_scratch/energy-slm-f16.gguf"

cfg = json.load(open(os.path.join(hf_dir, "config.json")))
tok_json = json.load(open(os.path.join(hf_dir, "tokenizer.json"), encoding="utf-8"))
weights = load_file(os.path.join(hf_dir, "model.safetensors"))

ARCHS = {"qwen3": gguf.MODEL_ARCH.QWEN3, "qwen2": gguf.MODEL_ARCH.QWEN2}
arch = gguf.MODEL_ARCH_NAMES[ARCHS[cfg["model_type"]]]
head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
w = gguf.GGUFWriter(out_path, arch)
w.add_name(sys.argv[3] if len(sys.argv) > 3 else "Smart-Home Energy SLM (Qwen3 architecture, from scratch)")
w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
w.add_context_length(cfg["max_position_embeddings"])
w.add_embedding_length(cfg["hidden_size"])
w.add_block_count(cfg["num_hidden_layers"])
w.add_feed_forward_length(cfg["intermediate_size"])
w.add_head_count(cfg["num_attention_heads"])
w.add_head_count_kv(cfg["num_key_value_heads"])
w.add_key_length(head_dim)
w.add_value_length(head_dim)
w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
w.add_rope_freq_base(cfg["rope_theta"] if "rope_theta" in cfg else cfg["rope_parameters"]["rope_theta"])

# Vocabulary: byte-level BPE tokens in id order, with added tokens marked CONTROL (special) or USER_DEFINED.
vocab = dict(tok_json["model"]["vocab"])
added = {t["id"]: t for t in tok_json["added_tokens"]}
for i, t in added.items():
    vocab[t["content"]] = i
n_vocab = cfg["vocab_size"]
id_to_tok = {i: s for s, i in vocab.items()}
tokens, types = [], []
for i in range(n_vocab):
    if i in id_to_tok:
        tokens.append(id_to_tok[i])
        if i in added:
            types.append(gguf.TokenType.CONTROL if added[i]["special"] else gguf.TokenType.USER_DEFINED)
        else:
            types.append(gguf.TokenType.NORMAL)
    else:
        tokens.append(f"[PAD{i}]")
        types.append(gguf.TokenType.UNUSED)
merges = [m if isinstance(m, str) else " ".join(m) for m in tok_json["model"]["merges"]]

w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("qwen2")
w.add_token_list(tokens)
w.add_token_types(types)
w.add_token_merges(merges)
w.add_eos_token_id(cfg["eos_token_id"])
w.add_pad_token_id(cfg["pad_token_id"] if cfg.get("pad_token_id") is not None else vocab["<|endoftext|>"])
w.add_add_bos_token(False)
chat_template = open(os.path.join(hf_dir, "chat_template.jinja")).read()
w.add_chat_template(chat_template)

names = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "self_attn.q_norm": "attn_q_norm",
    "self_attn.k_norm": "attn_k_norm",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
}


def add(name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float16 if arr.ndim == 2 else np.float32)
    w.add_tensor(name, arr)


add("token_embd.weight", weights.pop("model.embed_tokens.weight"))
add("output_norm.weight", weights.pop("model.norm.weight"))
if "lm_head.weight" in weights:                 # absent when the output layer is tied to the embeddings
    add("output.weight", weights.pop("lm_head.weight"))
for i in range(cfg["num_hidden_layers"]):
    for hf, gg in names.items():
        for kind in ("weight", "bias"):         # Qwen2 has q/k/v biases; Qwen3 has q/k norms instead
            key = f"model.layers.{i}.{hf}.{kind}"
            if key in weights:
                add(f"blk.{i}.{gg}.{kind}", weights.pop(key))
assert not weights, f"unmapped tensors: {sorted(weights)}"

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB)")
