# Smart-Home Appliance Energy SLM — Presentation

**Format:** 13 slides, about 12 minutes plus questions. Two presenters.

- **A = Ojas Chaudhary** (team leader): problem, data, training, results, demo
- **B = ____________** (teammate): persona, dataset design, limitations, business model, close

Each slide has what goes **on the slide**, then **Say:** notes for the speaker. Slides are separated by `---`.

---

## Slide 1 — Title  ·  *A · 30 s*

# Smart-Home Appliance Energy SLM
### A small language model that answers questions about a home's electricity use

Design Thinking Project · Team: Ojas Chaudhary (Leader), Parameswar Sahoo, Omsai, Nandhakishore

**Say:** "We built and trained a small language model that runs on a home energy hub and answers short questions about appliance electricity use, from 10 minutes to 24 hours ahead. It also knows when the data *can't* answer a question."

---

## Slide 2 — The problem  ·  *A · 1 min*

- Homeowners see raw meter numbers, not answers: *"Is tonight going to be a peak?"*
- **Peaks drive the bill.** The top 10% of readings hold 38% of all energy.
- Generic AI **guesses** — "your dishwasher used X" — when the meter can't tell appliances apart.
- Cloud AI means **uploading the family's daily routine**.

**Say:** "Three things go wrong today. Forecasts chase the average and miss peaks. Chatbots invent answers the data can't support. And sending readings to the cloud exposes when people are home or asleep. We wanted a model small enough to run on the hub itself."

---

## Slide 3 — The data  ·  *A · 1 min*

**UCI Appliances Energy Prediction** (Candanedo et al., 2017) · CC BY 4.0

| | |
|---|---|
| Readings | 19,735 × 10 minutes, one low-energy house in Belgium |
| Period | 11 Jan – 27 May 2016 |
| Channels | Appliance energy (Wh), lights, 9 rooms' temperature & humidity, outdoor weather |
| Shape | Median 60 Wh, max 1,080 Wh — heavily right-skewed |

**Traps in the data:** T6 is an *outdoor* sensor named like a room · rv1/rv2 are pure random noise · values are Wh per 10 min (60 Wh = 360 W), not kW

**Say:** "This dataset looks clean but has traps. A model that doesn't know T6 is outdoors, or that rv1 and rv2 are random, will give confident wrong explanations. We built those traps into our training questions."

---

## Slide 4 — Why a plain Transformer struggles  ·  *A · 1 min*

1. **Context window** — one day of readings as text is ~79,000 characters. A small model sees only a few hours, never "same time yesterday".
2. **Tokenisation & position** — numbers split into random fragments; nothing encodes hour-of-day or weekday.
3. **Training objective** — rewards fluent guesses over "cannot be determined", and pulls forecasts to the 40–60 Wh average.

**Say:** "These are the three Transformer weaknesses we identified on our Problem Spotter sheet. They shaped every design decision that follows."

---

## Slide 5 — User persona & design question  ·  *B · 1 min*

**Pieter Janssens, 31 — ML engineer at a home-energy-hub maker**
- Maintains the hub assistant; tests every release on 4,547 held-out questions
- *"It wasn't wrong — it never saw yesterday."*
- Real case: 26 May, flat 50–80 Wh, then **850 Wh at 16:40**. Yesterday jumped at the same time.

**How Might We** let a hub-sized model reach *"same time yesterday"* and *"same time last week"* without reading every hour in between?

**Say:** "Our persona is the engineer who owns this model. His biggest frustration is peaks the model couldn't see coming, because the warning sat a day back, outside its context. 64% of 'surprise' peaks had a high reading near the same time yesterday or last week. That became our How Might We statement."

---

## Slide 6 — Turning readings into training data  ·  *B · 1.5 min*

Raw CSV → **33,916 question → answer pairs**, 14 task types

| Answer the question | Know the limits |
|---|---|
| Forecast next 10 min / next hour | **Unanswerable** — occupancy, missing rooms |
| Classify level (low / moderate / high / peak) | **Units & cost** — no tariff in the data |
| Spike check, window stats, daily profile | **False premise** — "what caused the peak?" |
| Room comparison, lights, weather, time | **Noise features** — rv1/rv2, outdoor T6 |

Rules: every answer is **computed from the numbers in the prompt** · balanced yes/no · thresholds from training data only

**Say:** "We didn't use an LLM to write answers — every answer is calculated from the readings in the question, so the labels are always correct. A third of our task types exist to teach the model to say 'cannot be determined' instead of inventing something."

---

## Slide 7 — Splits & publishing  ·  *B · 45 s*

Split **by date**, so the model is always tested on the future:

| Split | Period | Rows |
|---|---|---|
| Train | up to 20 Apr 2016 | 25,027 |
| Validation | 20 Apr – 8 May | 4,342 |
| Test | after 8 May | 4,547 |

No context window crosses a split boundary.
Published: **huggingface.co/datasets/Design-Thinking-SLM/smart-home-energy-slm**

**Say:** "A random split would leak future readings into training. Splitting by date is how the model would really be used: trained on the past, asked about the future."

---

## Slide 8 — Training  ·  *A · 1 min*

| | |
|---|---|
| Base model | **Qwen2.5-0.5B-Instruct** (500M parameters, open weights) |
| Method | **LoRA** fine-tuning (r = 16, all linear layers) with TRL |
| Loss | On the answer only, not the question |
| Schedule | 2 epochs, 1,564 steps, effective batch 32, lr 2e-4 |
| Hardware | **One laptop GPU** (RTX 5060, 8 GB) · ~1.5 hours |

**Say:** "LoRA trains a small set of extra weights — a 45 MB file — instead of the whole model. That's why it trained on a laptop in an hour and a half, and why it could be updated per home later."

---

## Slide 9 — Results  ·  *A · 1.5 min*

**Validation loss kept falling — no overfitting**

| Step | 200 | 600 | 1,000 | 1,400 | 1,564 |
|---|---|---|---|---|---|
| Validation loss | 0.074 | 0.051 | 0.044 | 0.040 | **0.040** |
| Token accuracy | 97.5% | 98.0% | 98.3% | 98.5% | **98.5%** |

**Spot check, one test question per task: 10 / 14 exact**

- ✅ All "know the limits" tasks correct — cost, unanswerable, false premise, noise features
- ✅ Level classification, spike check, room comparison, lights
- ⚠️ Forecasts close: 600 vs 640 Wh, 90 vs 80 Wh
- ❌ Arithmetic and calendar: wrong 3-hour sum, wrong weekday

**Say:** "The most important result for us: it never invented an answer the data couldn't support. Its errors are exact sums and weekdays — a known weakness of small models, and easy to fix by computing those in code."

---

## Slide 10 — Live demo  ·  *A · 1.5 min*

```
python chat.py
```

1. `At 2016-05-26 17:10 the appliances used 620 Wh in 10 minutes. How would you classify this consumption level?` → **peak**
2. `The appliances used 70 Wh in 10 minutes. How much did that cost?` → **cannot be calculated: no tariff in the data**
3. `Appliance energy readings from 2016-05-14 13:50 to 2016-05-14 14:40 (10-minute intervals): 100, 220, 180, 120, 110, 90 (Wh, oldest first, 10-minute intervals). Which interval had the highest consumption?` → **220 Wh, at 14:10**

All three tested and answered correctly. Paste them exactly — the model was trained on this question style, and loosely worded versions (e.g. "Readings: 100, 220, … Wh") can give odd answers.

**Say:** "A correct answer, an honest refusal, and a correct lookup. This is the model we trained, running locally — no internet needed after the first download."

*(Backup if the demo fails: show the spot-check outputs from slide 9.)*

---

## Slide 11 — Limitations & next steps  ·  *B · 1 min*

**Limitations**
- Exact arithmetic and weekdays are unreliable → compute in code, let the model explain
- One house, January–May only: no summer data
- Full test-set evaluation (4,547 questions) and the safety prompt set still to run
- Prompts hold ~1 hour of history; our HMW's "yesterday / last week" context is the next build

**Next steps**
1. Per-task test scores and the positive/negative (security) prompt evaluation
2. Add same-time-yesterday and last-week readings to the prompt (HMW 1)
3. 4-bit compression to fit the hub; retrain with summer data

**Say:** "We want to be clear about what this model is and isn't yet. The trained model is the baseline; adding lagged context from our How Might We is the next step."

---

## Slide 12 — Who would use it  ·  *B · 1 min*

| | |
|---|---|
| **Customers** | Homeowners on smart meters or peak-based tariffs · energy advisers · hub makers |
| **Value** | Peak warnings before the evening · honest answers · correct units · **private: runs on the hub** |
| **Channels** | Pre-installed on energy hubs · hub app alerts · adviser home audits |
| **Costs** | A few GPU-hours per training run · free open-weight base · no cloud or API fees |
| **Impact** | kWh moved out of peaks · peak recall · honesty rate · zero data uploaded |

**Say:** "Because it runs on hardware the household already owns, there's no per-question cloud cost and no privacy trade-off. Success is measured in peaks avoided, not just accuracy."

---

## Slide 13 — Thank you  ·  *B · 15 s*

- **Code:** github.com/Ojas1004/design_thinking_slm_project
- **Dataset:** huggingface.co/datasets/Design-Thinking-SLM/smart-home-energy-slm
- **Model:** huggingface.co/Design-Thinking-SLM/smart-home-energy-slm

### Questions?

---

## Backup — Likely faculty questions

**Why a 0.5B model, not a bigger one?**
It has to run on a home hub with no cloud. 0.5B fits in a few hundred MB after 4-bit compression, and trained on one laptop GPU.

**How do you know the answers in your dataset are correct?**
They're computed by code from the readings in each prompt, not written by a person or an LLM. Thresholds (low/moderate/high/peak) come from the training period only.

**Isn't 98.5% token accuracy misleading?**
Yes, partly — most answer tokens are fixed wording. That's why we spot-checked whole answers (10/14 exact) and plan a full per-task test evaluation.

**Why split by date instead of randomly?**
Neighbouring 10-minute readings are almost identical. A random split leaks them between train and test and inflates scores.

**What is LoRA?**
Instead of updating all 500M weights, it trains small low-rank matrices added to each layer — about 45 MB — so training is fast and cheap, and the base model is unchanged.

**What does "cannot be determined" protect against?**
The meter measures only total energy. Without it, the model would invent which appliance ran or whether someone was home — wrong answers, and a privacy risk.

**What about prompt injection or misuse (e.g. "when is the house empty?")?**
We wrote a positive/negative prompt set (12 legitimate, including hard positives, and adversarial prompts mapped to the OWASP LLM Top 10). Evaluating the model on it is our next step.

**Did you implement your How Might We?**
Not yet in training. The current model is the fine-tuned baseline; adding same-time-yesterday and last-week readings to the context is the planned next version, and this baseline is what it will be compared against.
