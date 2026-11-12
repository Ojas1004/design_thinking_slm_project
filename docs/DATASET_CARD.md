---
license: cc-by-4.0
language:
- en
task_categories:
- text-generation
- question-answering
tags:
- energy
- smart-home
- time-series
- forecasting
- slm
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# Smart-Home Appliance Energy — SLM Instruction Data

Prompt/completion pairs for fine-tuning a small language model that runs on a home energy hub and answers short-horizon (10 min – 24 h) questions about appliance electricity use.

## Source

Built from the UCI *Appliances Energy Prediction* dataset (Candanedo, Feldheim & Deramaix, *Energy and Buildings* 140, 2017): 10-minute readings from one house, 2016-01-11 to 2016-05-27.

## Splits (chronological, no overlap)

| Split | Period | Rows |
|---|---|---|
| train | up to 2016-04-20 | 25,027 |
| validation | 2016-04-20 – 2016-05-08 | 4,342 |
| test | after 2016-05-08 | 4,547 |

Context windows never cross a split boundary or a gap in the 10-minute grid, and no prompt appears in more than one split.

## Fields

- `prompt` — question with the evidence (readings) included
- `completion` — target answer, computed only from evidence in the prompt
- `task` — one of 14 task families (forecasting, peak/spike detection, comparisons, unanswerable questions, trap questions)
- `split` — train / val / test
- `answerable` — `false` for questions the data cannot answer (occupancy, per-appliance use, tariffs, faults); the completion says so instead of inventing a value
- `label` — short categorical label for scoring

## Notes

- Values are Wh per 10 minutes (60 Wh = 360 W average).
- Level/spike thresholds come from train-split percentiles only.
- Yes/no tasks are balanced 50/50.
