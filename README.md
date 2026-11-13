
# Teammates 
# Ojas Chaudhary (Leader)
# Parameswar Sahoo
# Omsai
# Nandhakishore


# Smart-Home Appliance Energy SLM

A small language model for a home energy hub that answers short-horizon (10 min – 24 h) questions about appliance electricity use, trained on instruction data built from the UCI *Appliances Energy Prediction* dataset (Candanedo et al., 2017).

**Processed dataset (train / validation / test):** https://huggingface.co/datasets/Design-Thinking-SLM/smart-home-energy-slm

**Trained model (LoRA adapter on Qwen2.5-0.5B-Instruct):** https://huggingface.co/Design-Thinking-SLM/smart-home-energy-slm

## Repository layout

| Path | Contents |
|---|---|
| `data/energydata_complete.csv` | Raw UCI data: 10-minute readings, 2016-01-11 to 2016-05-27 |
| `notebooks/energydata_to_sml_jsonl.ipynb` | Current generator: builds prompt/completion pairs, chronological splits (stdlib only) |
| `notebooks/data_conversion.ipynb` | First, superseded pipeline (kept for comparison) |
| `convert_to_hf.py` | Converts the generated JSONL splits into Hugging Face Parquet layout |
| `train.py` | LoRA fine-tune of Qwen2.5-0.5B-Instruct on the dataset (`--smoke` for a 20-step trial) |
| `data/positive_negative_prompts.jsonl` | Positive / negative prompt set for evaluating the model |
| `docs/` | Design-thinking sheets (problem framing, persona, HMW, business model canvas) and the dataset card |

## Data pipeline

1. Run `notebooks/energydata_to_sml_jsonl.ipynb` → `energydata_sml_{train,val,test}.jsonl` (seed 42, 2,500 train examples per task).
2. `python convert_to_hf.py <jsonl dir> hf_dataset` → Parquet splits.
3. Upload `hf_dataset/` to the Hugging Face Hub.

| Split | Period | Rows |
|---|---|---|
| train | up to 2016-04-20 | 25,027 |
| validation | 2016-04-20 – 2016-05-08 | 4,342 |
| test | after 2016-05-08 | 4,547 |

Key design choices: chronological split with no context window crossing a boundary; thresholds from the train split only; completions computed only from evidence in the prompt; balanced yes/no tasks; explicit "cannot be determined" answers for occupancy, per-appliance use, tariffs and faults; trap questions on misleading channels (T6, rv1/rv2, Wh vs kW).

## Training

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers datasets peft trl accelerate
python train.py        # ~1.5 h on an 8 GB RTX 5060 Laptop GPU
```

LoRA r=16 on all linear layers, 2 epochs (1,564 steps, effective batch 32), lr 2e-4 cosine, loss on completions only. Final validation loss 0.040, token accuracy 98.5%. Output: `out/qwen0.5b-energy-lora/final`. Known weakness: exact arithmetic and weekday lookups, which a deployment should compute in code.

## Data licence

UCI Appliances Energy Prediction dataset — CC BY 4.0. Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81–97.
