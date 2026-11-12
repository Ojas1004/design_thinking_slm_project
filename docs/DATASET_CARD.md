---
pretty_name: Smart-Home Appliance Energy SLM
annotations_creators:
- machine-generated
language_creators:
- machine-generated
language:
- en
license: cc-by-4.0
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- extended|other-uci-appliances-energy-prediction
task_categories:
- text-generation
- question-answering
- time-series-forecasting
task_ids:
- open-domain-qa
- closed-domain-qa
tags:
- energy
- smart-home
- forecasting
- instruction-tuning
- slm
dataset_info:
  features:
  - name: prompt
    dtype: string
  - name: completion
    dtype: string
  - name: task
    dtype:
      class_label:
        names:
          '0': daily_profile
          '1': false_premise
          '2': forecast_hour
          '3': forecast_next
          '4': level_classify
          '5': lights_activity
          '6': noise_features
          '7': outdoor_weather
          '8': room_compare
          '9': spike_check
          '10': time_features
          '11': unanswerable
          '12': units_cost
          '13': window_stats
  - name: answerable
    dtype: bool
  - name: label
    dtype:
      class_label:
        names:
          '0': answerable
          '1': high
          '2': low
          '3': moderate
          '4': 'no'
          '5': none
          '6': peak
          '7': premise_false
          '8': premise_true
          '9': unanswerable
          '10': 'yes'
  splits:
  - name: train
    num_bytes: 1306424
    num_examples: 25027
  - name: validation
    num_bytes: 238262
    num_examples: 4342
  - name: test
    num_bytes: 243747
    num_examples: 4547
  download_size: 1788433
  dataset_size: 1788433
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

# Dataset Card for "Smart-Home Appliance Energy SLM"

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks](#supported-tasks)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
- [Considerations for Using the Data](#considerations-for-using-the-data)
- [Additional Information](#additional-information)

## Dataset Description

- **Source data:** [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
- **Source paper:** Candanedo, Feldheim & Deramaix (2017), [*Energy and Buildings* 140, 81–97](https://doi.org/10.1016/j.enbuild.2017.01.083)
- **Size of dataset files:** 1.79 MB (Parquet)
- **Number of rows:** 33,916

### Dataset Summary

Instruction (prompt → completion) pairs for fine-tuning a **small language model that runs on a home energy hub**. The model answers short-horizon (10 min – 24 h) questions about one house's appliance electricity use: forecasting the next reading or hour, classifying load levels, spotting spikes, comparing rooms, and converting units and costs.

The data also teaches the model **what it cannot know**. The meter records only total appliance energy, so questions about occupancy, individual appliances, tariffs or faults get an explicit "cannot be determined" answer, and trap questions test misleading channels (an outdoor sensor named like a room, random-noise columns, Wh vs kW).

### Supported Tasks

- `text-generation` / instruction tuning: train an SLM to produce the `completion` for each `prompt`.
- Classification-style evaluation: compare the model's answer against the short `label` (e.g. `low` / `moderate` / `high` / `peak`, `yes` / `no`, `unanswerable`).
- Abstention evaluation: the `answerable` flag measures whether the model refuses to invent answers the data cannot support.

### Languages

English (`en`).

## Dataset Structure

### Data Instances

A forecasting example from `train`:

```json
{
  "prompt": "Short-term load forecast. History (10-min steps, Wh): \n22:00  90 Wh\n22:10  60 Wh\n22:20  60 Wh\n22:30  50 Wh\n22:40  60 Wh\n22:50  50 Wh. Predict the next value.",
  "completion": "The next 10-minute reading is expected to be about 60 Wh, a low level for this house. This is a point forecast from the recent readings and it carries uncertainty: appliance load in this dataset is spiky and right-skewed (median ≈ 60 Wh).",
  "task": "forecast_next",
  "answerable": true,
  "label": "low"
}
```

An unanswerable example:

```json
{
  "prompt": "The appliances used 60 Wh at 2016-02-12 02:20. Which appliance was running?",
  "completion": "That cannot be determined. 60 Wh is a single aggregate meter reading for all appliances. The dataset has no per-appliance sub-metering, so separating it into individual devices (load disaggregation) is not possible without device-level data or signatures.",
  "task": "unanswerable",
  "answerable": false,
  "label": "unanswerable"
}
```

### Data Fields

| Column | Type | Description |
|---|---|---|
| `prompt` | string | Question, with the evidence (readings) included |
| `completion` | string | Target answer, computed only from evidence in the prompt |
| `task` | class label (14) | Task family (see below) |
| `answerable` | bool | `false` when the data cannot answer the question; the completion says so instead of inventing a value |
| `label` | class label (11) | Short answer for scoring: `low` / `moderate` / `high` / `peak`, `yes` / `no`, `premise_true` / `premise_false`, `answerable` / `unanswerable`, or `none` for free-text answers |

Task families: `forecast_next`, `forecast_hour`, `window_stats`, `daily_profile`, `level_classify`, `spike_check`, `room_compare`, `lights_activity`, `outdoor_weather`, `time_features`, `units_cost`, `noise_features`, `false_premise`, `unanswerable`.

### Data Splits

The split is **chronological**, so the test set is genuinely in the model's future.

| Split | Period | Rows |
|---|---|---|
| train | 2016-01-11 – 2016-04-20 | 25,027 |
| validation | 2016-04-20 – 2016-05-08 | 4,342 |
| test | 2016-05-08 – 2016-05-27 | 4,547 |

No context window crosses a split boundary or a gap in the 10-minute grid, and no prompt appears in more than one split.

## Dataset Creation

Generated by a stdlib-only Python notebook from the raw UCI CSV (19,735 ten-minute readings from one low-energy house in Belgium):

- Level and spike thresholds are percentiles of the **train split only**, so no information leaks from validation/test.
- Every completion is computed from evidence present in the prompt.
- Yes/no tasks are sampled 50/50.
- Prompts are de-duplicated corpus-wide; any prompt that would map to two different completions is dropped.
- Seed 42; 2,500 training examples per task family before de-duplication.

## Considerations for Using the Data

- **One house, winter–spring only.** Patterns will not transfer directly to other homes or to summer.
- **Units:** values are Wh per 10 minutes (60 Wh = 360 W average), not kW.
- **Heavy-tailed target:** median ≈ 60 Wh, max 1,080 Wh; the top 10% of readings hold 38% of the energy.
- `T6` is an outdoor sensor (r = 0.97 with `T_out`), and `rv1` / `rv2` are random noise columns in the source data.

## Additional Information

### Licensing Information

CC BY 4.0, following the source UCI dataset.

### Citation Information

```bibtex
@article{candanedo2017appliances,
  title   = {Data driven prediction models of energy use of appliances in a low-energy house},
  author  = {Candanedo, Luis M. and Feldheim, V{\'e}ronique and Deramaix, Dominique},
  journal = {Energy and Buildings},
  volume  = {140},
  pages   = {81--97},
  year    = {2017},
  doi     = {10.1016/j.enbuild.2017.01.083}
}
```

### Usage

```python
from datasets import load_dataset

ds = load_dataset("Ojas1004/smart-home-energy-slm")
print(ds["train"][0])
```
