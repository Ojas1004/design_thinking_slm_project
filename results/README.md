# Results of the from-scratch Qwen3 models

The trained weights are on Hugging Face (`Design-Thinking-SLM/smart-home-energy-qwen3-scratch`). This folder keeps
the small files that describe each run, so the article's numbers can be checked without retraining.

| Folder | Run |
|---|---|
| `qwen3_scratch/` | Final model: 109M parameters, trained on `energydata_sml_large` (UCI + 20 REFIT homes, 148,363 training pairs) |
| `qwen3_scratch_v1/` | First run: same architecture, trained on `energydata_sml` (UCI house only); overfitted |

Each folder has:

- `config.json`: architecture, parameter count, tokenizer, prompt format and training hyperparameters
- `loss_curve.png`: training and validation loss
- `test_eval.json`: the notebook's own test-set evaluation
- `eval_full.jsonl`: every answer from `scripts/evaluate_full.py` (up to 200 test questions per task and source,
  same seed for both runs, so the comparison is paired)

To recompute the statistics, tables and figures (written to `paper/`, which git ignores) from these answers without retraining:

```bash
mkdir -p out/qwen3_scratch out/qwen3_scratch_v1
cp results/qwen3_scratch/eval_full.jsonl    out/qwen3_scratch/
cp results/qwen3_scratch_v1/eval_full.jsonl out/qwen3_scratch_v1/
python scripts/analyze_eval.py
```
