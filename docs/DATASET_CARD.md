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

| Column | Type | Description |
|---|---|---|
| `prompt` | string | Question, with the evidence (readings) included |
| `completion` | string | Target answer, computed only from evidence in the prompt |
| `task` | class label (14) | Task family: `forecast_next`, `forecast_hour`, `window_stats`, `daily_profile`, `level_classify`, `spike_check`, `room_compare`, `lights_activity`, `outdoor_weather`, `time_features`, `units_cost`, `noise_features`, `false_premise`, `unanswerable` |
| `answerable` | bool | `false` for questions the data cannot answer (occupancy, per-appliance use, tariffs, faults); the completion says so instead of inventing a value |
| `label` | class label (11) | Short answer label for scoring: `low` / `moderate` / `high` / `peak`, `yes` / `no`, `premise_true` / `premise_false`, `answerable` / `unanswerable`, or `none` for free-text answers |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("Ojas1004/smart-home-energy-slm")
print(ds["train"][0])
```

## Notes

- Values are Wh per 10 minutes (60 Wh = 360 W average).
- Level/spike thresholds come from train-split percentiles only.
- Yes/no tasks are balanced 50/50.
