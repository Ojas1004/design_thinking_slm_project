# Smart-Home Appliance Energy SLM

**Team:** Ojas Chaudhary (Leader), Parameswar Sahoo, Omsai, Nandhakishore. Jain University, Design Thinking course
(25ESHS01), guided by Dr. Sujeet S. Jagtap.

Small language models for a home energy hub. They answer short questions about household electricity readings
(costs, load levels, spikes, short-horizon forecasts) and say **"cannot be determined"** when the meter data cannot
answer: occupancy, which appliance was running, faults, tariffs.

This repository covers the whole project: data collection, question–answer generation, training, evaluation,
the research article, and the chat demo in Ollama.

| | Link |
|---|---|
| Dataset (`default`: UCI house, 33,916 pairs; `large`: UCI + 20 REFIT homes, 200,335 pairs) | https://huggingface.co/datasets/Design-Thinking-SLM/smart-home-energy-slm |
| Model 1: Qwen3 architecture trained **from scratch** (109M), with GGUF for Ollama | https://huggingface.co/Design-Thinking-SLM/smart-home-energy-qwen3-scratch |
| Model 2: **LoRA** on Qwen2.5-0.5B-Instruct, with GGUF for Ollama | https://huggingface.co/Design-Thinking-SLM/smart-home-energy-slm |

## Results at a glance

From-scratch model, 3,956 held-out test questions (statistical tests from `scripts/analyze_eval.py`):

| | First version (UCI corpus only) | Final version (UCI + REFIT) |
|---|---|---|
| Exact match, UCI house | 61.2% | **69.0%** (95% CI 67.1–70.9) |
| Exact match, REFIT homes | 7.8% | **43.1%** (95% CI 39.7–46.6) |
| Unanswerable questions answered anyway | 0 of 364 | **0 of 364** |
| Answerable questions refused | 0 | **0 of 3,592** |

- **Errors:** wrong answers are mostly arithmetic. 82% have the right wording with a wrong number.
- **Forecasts:** no better than repeating the last reading.
- **Speed:** in Ollama the model answers in 0.6 s on a laptop CPU.

## Limitations of the from-scratch Qwen3 model

**Language**
- **It only understands the dataset's question formats.** It started from random weights and saw only 148,363
  generated question–answer pairs, so it has never read ordinary English. Casual questions such as "current
  temperature is 32, how will it influence?" produce broken text. Use the LoRA model (`energy-lora`) for those.
- **One question at a time.** It was trained on single questions with a 448-token limit, so it cannot hold a
  conversation; the Ollama recipe sends only the latest message.

**Answers**
- **Arithmetic is unreliable.** 82% of its wrong answers have the right sentence with a wrong number.
  - Accuracy falls as the question holds more numbers: on window statistics, from 54% (fewer than 20 numbers) to
    26% (40 or more).
  - Sums over a day of hourly values are right only 5% (UCI) to 15% (REFIT) of the time.
  - Cost questions are reliable only with the exact numbers seen in tests. With new prices it multiplied wrongly
    (0.120 × €0.30 = €0.0376 instead of €0.0360).
- **Values that must be worked out can be wrong:** the weekday of a date (88% right for the UCI house, 62% for
  REFIT homes) and the warmest or coldest room (47%).
- **Forecasts are no better than repeating the last reading.** In none of the four forecast comparisons does the
  95% bootstrap interval show it beating that baseline. For the next REFIT reading it is slightly worse (+6.0 Wh,
  95% CI +0.4 to +12.1).
  It sees only the last 3–12 readings, not the same time yesterday or last week.

**Scope**
- **Narrow data.** Room, weather and lighting questions come from one house in Belgium, January–May 2016. REFIT
  adds 20 UK homes, but only whole-house load. "High" and "peak" thresholds are specific to each house.
- **Inefficient size.** 77.7M of its 109M parameters are the embedding table that comes with the 151,669-token
  Qwen3 tokenizer, which the course notebook uses. That tokenizer also splits every number into single digits,
  which makes arithmetic harder.

**Evaluation**
- **The test checks new readings, not new wording.** Test questions use the same templates as training, on later
  dates.
- **Scoring is strict.** Exact match counts an answer off by 10 Wh as wrong.
- **One run only.** Each model was trained once, with one seed.

It is a course prototype, not a tool for billing, forecasting or safety decisions. A deployment should compute
sums and dates in code and let the model explain the result.

## Repository layout

| Path | Contents |
|---|---|
| `data/energydata_complete.csv` | Raw UCI data: one house in Belgium, 10-minute readings, 2016-01-11 to 2016-05-27 |
| `notebooks/energydata_to_sml_jsonl.ipynb` | **Step 2.** Generator: question–answer pairs from UCI (+ REFIT), chronological splits |
| `notebooks/5_slm_from_scratch_qwen.ipynb` | **Step 3a.** The course reference notebook adapted to our data: Qwen3 architecture trained from scratch |
| `notebooks/reference/` | The unmodified course notebook, and the first from-scratch run (UCI corpus only) |
| `train.py` | **Step 3b.** LoRA fine-tune of Qwen2.5-0.5B-Instruct (`--smoke` for a 20-step trial) |
| `scripts/evaluate_full.py`, `scripts/analyze_eval.py` | **Step 4.** Large test-set evaluation and statistics (Wilson CIs, McNemar, Wilcoxon, bootstrap) |
| `scripts/export_qwen3_to_hf.py`, `scripts/merge_lora.py`, `scripts/export_hf_to_gguf.py` | **Step 5.** Convert both models to GGUF for Ollama |
| `Modelfile`, `Modelfile.lora` | **Step 6.** Ollama recipes for the two models |
| `scripts/check_ollama.py` | Asks one test question per task through Ollama and compares with the reference |
| `docs/demo_questions.md` | Questions to paste into the chat demo, with the answers the model gives |
| `docs/` | Design-thinking sheets: problem framing, persona, How-Might-We, business model canvas |
| `convert_to_hf.py`, `chat.py` | Parquet conversion of the original corpus; terminal chat with the LoRA adapter |
| `notebooks/data_conversion.ipynb` | First, superseded pipeline (kept for comparison) |

Run every command below from the repository root. Generated data and models go to `data/` and `out/`, which git
ignores.

## Step 0: Environment

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate      # fish: source .venv/bin/activate.fish
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

All training ran on one laptop: RTX 5060 Laptop GPU (8 GB), Ryzen 7 250, 14 GB RAM.

## Step 1: Data collection

| Source | What | Licence |
|---|---|---|
| [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) (Candanedo et al. 2017) | One house, 19,735 ten-minute readings: appliance energy (Wh), lights, 9 room temperature/humidity pairs, weather | CC BY 4.0 |
| [REFIT](https://doi.org/10.1038/sdata.2016.122) (Murray et al. 2017), via [`Jainam03/RESCAST-100k-REFIT`](https://huggingface.co/datasets/Jainam03/RESCAST-100k-REFIT) | 20 UK homes, 1,044,201 fifteen-minute whole-house readings, 2013–2015 | CC BY 4.0 |

The UCI file is in `data/`. The generator downloads REFIT the first time it runs, into `data/refit_data/`.

The exploratory analysis behind the design (skewed load, autocorrelation, misleading sensor columns) is in
`docs/problem_framing_answers.md` and the article's Data section.

## Step 2: Generate the question–answer corpus

```bash
cd notebooks && jupyter nbconvert --to notebook --execute --inplace energydata_to_sml_jsonl.ipynb && cd ..
```

This writes `data/energydata_sml_large{,_train,_val,_test}.jsonl`: 200,335 pairs over 14 task families
(148,363 train / 23,734 validation / 28,238 test). Generation is seeded: a rerun reproduces the published files
byte for byte. The design rules, each one a fix for a problem found in the data, are:

- **Chronological split.** UCI: train to 2016-04-20, validation to 2016-05-08. REFIT: train to 2015-01-31,
  validation to 2015-03-31. No context window crosses a split or a gap in the time grid.
- **Answers come from the prompt.** Every answer is computed by code from numbers in the question.
- **No leakage.** Level and spike thresholds are percentiles of the training split only.
- **Refusals and traps.** Explicit "cannot be determined" questions (occupancy, appliances, faults, tariffs) and
  false-premise questions. Trap questions cover `T6` (an outdoor sensor), `rv1`/`rv2` (random noise) and Wh vs kW.
- **Balanced and unique.** Yes/no questions are balanced 50/50, prompts are unique, and an assertion checks that
  no prompt appears in two splits.

The smaller original corpus is the dataset's `default` configuration, and the LoRA model was trained on it. To
regenerate it, set `OUTPUT_STEM = 'energydata_sml'`, `EXAMPLES_PER_TASK_TRAIN = 2500` and `INCLUDE_REFIT = False`
in the notebook's first code cell.

## Step 3a: Train the from-scratch model (course notebook)

```bash
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 5_slm_from_scratch_qwen.ipynb && cd ..
```

This trains a 109M-parameter Qwen3-architecture decoder from random weights: 8 layers, width 512, grouped-query
attention, the Qwen3-8B tokenizer. Loss is on answer tokens only, 72,000 steps of batch 8. It takes about 97
minutes and uses at most 4.6 GB of GPU memory.

The notebook's first cell lists every change from the course reference notebook (`notebooks/reference/`). It
writes `out/qwen3_scratch/` (the checkpoint, loss curve, test scores, and an `hf_export/` folder). Set
`PUSH_TO_HUB = True` in its last cell to upload.

## Step 3b: Train the LoRA model

```bash
python train.py --smoke      # optional 20-step check
python train.py              # about 1.5 h
```

This is a LoRA fine-tune of Qwen2.5-0.5B-Instruct (r = 16 on all linear layers, 2 epochs, lr 2e-4) on the `default`
configuration. It writes `out/qwen0.5b-energy-lora/final` and reaches validation loss 0.040 and token accuracy
98.5%. Because it starts from an instruction-tuned model, it also handles casually worded questions.

## Step 4: Evaluate

```bash
python scripts/export_qwen3_to_hf.py                        # needed by the evaluation (and by Step 5)
python scripts/evaluate_full.py out/qwen3_scratch/hf_qwen3 out/qwen3_scratch/eval_full.jsonl 200
python scripts/analyze_eval.py                              # -> paper/results.json, tables and figures (not tracked)
python scripts/make_article_figures.py                      # data and loss figures
```

`evaluate_full.py` asks up to 200 test questions per (task, source) group with greedy decoding. `analyze_eval.py`
compares two runs on the same questions. It expects the first (UCI-only) model's answers in
`out/qwen3_scratch_v1/eval_full.jsonl`; to get them, train the notebook with `DATA_STEM = "energydata_sml"` and
evaluate it the same way.

## Step 5: Convert to GGUF for Ollama

Ollama runs models in the GGUF format. `export_hf_to_gguf.py` writes it directly with the `gguf` package, because
Ollama 0.40 on Linux cannot import safetensors folders.

```bash
# from-scratch model: notebook checkpoint -> Hugging Face Qwen3 -> GGUF (379 MB)
python scripts/export_qwen3_to_hf.py
python scripts/export_hf_to_gguf.py

# LoRA model: merge the adapter into Qwen2.5-0.5B-Instruct -> GGUF (994 MB)
python scripts/merge_lora.py
python scripts/export_hf_to_gguf.py out/lora_merged out/lora/energy-lora-f16.gguf "Smart-Home Energy SLM (Qwen2.5-0.5B-Instruct + LoRA)"
```

The from-scratch model scales its token embeddings by √512, which Hugging Face Qwen3 does not. The converter
folds that scale into the input embedding; the converted model's logits match the original to within 1.5 × 10⁻⁵.

Instead of converting, you can download the ready GGUF files from the two model repositories above.

## Step 6: Chat demo in Ollama

Install Ollama (Arch/CachyOS: `sudo pacman -S ollama-cuda && sudo systemctl start ollama`), then:

```bash
ollama create energy-slm  -f Modelfile         # from-scratch model
ollama create energy-lora -f Modelfile.lora    # LoRA model
ollama run energy-lora
```

| Model | Best for |
|---|---|
| `energy-lora` | Casually worded questions ("My appliances used 450 Wh in the last 10 minutes. Is that normal?") |
| `energy-slm` | Questions phrased like the dataset. It was trained from scratch, so it knows no other English. |

Questions known to work, with the answers they get, are in [`docs/demo_questions.md`](docs/demo_questions.md).
Paste one question at a time; wrap multi-line questions in `"""`. Both Modelfiles decode greedily and send only
the latest message, because the models were trained on single questions. `Modelfile.lora` adds a light
repetition penalty, which stops greedy decoding from looping on unusual questions.

`python scripts/check_ollama.py energy-lora` asks one test question per task through Ollama's API.

## Data licence and citations

The data is CC BY 4.0, following the sources:

- Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances
  in a low-energy house. *Energy and Buildings*, 140, 81–97.
- Murray, D., Stankovic, L., & Stankovic, V. (2017). An electrical load measurements dataset of United Kingdom
  households from a two-year longitudinal study. *Scientific Data*, 4, 160122.
