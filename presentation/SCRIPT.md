# Presentation script — Smart-Home Appliance Energy SLM

**Two speakers · 15 slides · about 13 minutes + questions, including a live Ollama demo**

- **A = Ojas**: opening, the data, the two models, results, Ollama and the live demo
- **B = teammate**: limitations, where the model fails, positive/negative prompts, closing

Handovers happen 5 times (after slides 4, 6, 9, 10 and 12). Each is marked **→ HANDOVER**, with a one-line bridge.

Slides 16–19 are the appendix (full prompt tables). Don't present them; open them only if faculty asks about a
specific prompt.

---

## Slide 1 — Title · A · 30 s

> Good morning. We're presenting our Design Thinking project: small language models that answer questions about a home's electricity use, and run offline on the home's own hub.
>
> Our team is Ojas, Parameswar, Omsai and Nandhakishore. I'm Ojas, and I'll be presenting with [teammate's name].
>
> We'll cover our data and its limitations, the two models we trained, how we tested them, and then we'll show them running live in Ollama.

---

## Slide 2 — The dataset · A · 45 s

> Our main data is the UCI Appliances Energy Prediction dataset, published by Candanedo and colleagues in 2017.
>
> It's from one low-energy house in Belgium, recorded every ten minutes from January to May 2016: almost twenty thousand readings. Each row has the energy used by all appliances, the lights, temperature and humidity from nine sensors, and the outdoor weather.
>
> One important point: the meter only records the **total** for all appliances. It can't tell us which appliance was on, whether anyone was home, or what the electricity cost. That shapes everything that follows.

---

## Slide 3 — What we built · A · 1 min

> A language model can't learn from a raw table, so we turned the readings into questions and answers.
>
> Here's an example: we give it an hour of readings and ask for the next value. The answer gives a number and a level, like "low for this house". **Every answer is computed by code** from the numbers in the question, so the labels are always correct.
>
> Our first version had about 34,000 pairs from the one house. One house is a narrow base, so we added a second public dataset, **REFIT: twenty homes in the UK**, recorded every fifteen minutes. The chart shows the result: **over 200,000 questions** across 14 task types, blue from the Belgian house, orange from the UK homes.
>
> We split by date, not randomly, so the model is always tested on future readings it has never seen.

---

## Slide 4 — Why this dataset · A · 45 s

> Why this data? First, it's a **real household routine** at ten-minute resolution. On the left: quiet nights, a morning rise, and a clear evening peak at six o'clock. That's exactly the kind of "next ten minutes" question a home hub should answer.
>
> Second, **peaks matter**. The median reading is 60 watt-hours, but the top 10% of readings hold 37% of all the energy.
>
> Third, it **tests honesty**: the data can't say which appliance ran, so it shows whether a model admits what it doesn't know.
>
> But the dataset also has real limitations, which [teammate] will explain.

**→ HANDOVER to B**

---

## Slide 5 — Limitations · B · 45 s

> Thanks, Ojas. Three kinds of limitations.
>
> First, **narrow scope**: one house, January to May only, under twenty thousand rows. We've fixed the one-house part with the REFIT homes.
>
> Second, **missing labels**: no per-appliance data, no occupancy, no electricity price, no fault labels.
>
> Third, **traps**. A sensor called T6 looks like a room, but it's actually outside the building. Two columns, rv1 and rv2, are pure random numbers. And the values are watt-hours per ten minutes, so 60 means 360 watts, not 60 kilowatts.

---

## Slide 6 — How we designed around them · B · 45 s

> We built the dataset around these limitations.
>
> For missing labels, the model answers "cannot be determined" and explains why. For the traps, we wrote questions whose correct answer excludes T6 or says rv1 and rv2 carry no information. We added unit questions and level bands for peaks, computed every threshold from the training period only, and added the twenty UK homes, each with its own thresholds.
>
> Without this, a model would guess these answers fluently, and wrongly. We taught it what it **cannot** know.
>
> Now Ojas will explain the models.

**→ HANDOVER to A**

---

## Slide 7 — Two models · A · 1 min

> We trained two models on this data, both on one laptop GPU.
>
> The first is a **LoRA** model. It starts from Qwen 2.5, a small pretrained model of half a billion parameters that already understands English, and trains a 45-megabyte add-on on our first dataset.
>
> The second follows the **course notebook**: a model with the Qwen3 architecture, 109 million parameters, trained **from random weights**. It borrows only Qwen3's tokenizer, so everything it knows comes from our 200,000 questions. It trained in 97 minutes.
>
> Both are small enough to run on a home hub, and both end up in Ollama, as we'll show.

---

## Slide 8 — Training results · A · 45 s

> On the left, the LoRA model: validation loss fell from 0.074 to 0.040 and was still falling, and 98.5% of answer tokens were correct.
>
> On the right, the from-scratch model. Our first run used only the Belgian house, and it **overfitted**: training loss kept falling, but validation loss stalled. With six times more data, the two curves stay together. That's the main reason we added the REFIT homes.

---

## Slide 9 — Test results · A · 1 min

> We then tested the from-scratch model on almost four thousand questions it had never seen, comparing every answer word for word with the correct one.
>
> The most important result: on 364 questions the data cannot answer, it made up an answer **zero times**, and it never refused a legitimate question.
>
> Overall, exactly-right answers rose from 61 to 69 percent on the Belgian house, and from 8 to 43 percent on the UK homes, compared with our first run on the same questions. That improvement is statistically significant. The refusal, false-premise and noise-column tasks are at 95 to 100 percent.
>
> But it isn't perfect, and [teammate] will show where it fails.

**→ HANDOVER to B**

---

## Slide 10 — Where it still fails · B · 1 min

> Its main weakness is **arithmetic**. 82% of its wrong answers have the right sentence with a wrong number. The chart shows why: the more numbers in the question, the more mistakes. Sums over a whole day are right only 5 to 15 percent of the time.
>
> Its **forecasts** are about as good as simply repeating the last reading, no better.
>
> And because the from-scratch model never read ordinary English, a casual question gets broken text. The LoRA model handles those.
>
> Our fixes: let code do the sums and dates while the model explains them, and give it readings from the same time yesterday and last week, which was our How-Might-We idea from the design phase.
>
> Ojas will now show both models running live.

**→ HANDOVER to A**

---

## Slide 11 — Ollama · A · 45 s

> To run on a home hub, the model has to work offline. We converted both models into GGUF, the file format Ollama uses: 379 megabytes for the from-scratch model and under a gigabyte for LoRA. We checked the conversion: the outputs match the trained model.
>
> They answer in about **0.6 seconds on a laptop CPU**, with no internet. We use `energy-lora` for everyday questions and `energy-slm` for questions in the dataset's format.

---

## Slide 12 — Live demo · A · 1.5 min

*Switch to the terminal. If anything goes wrong, this slide shows the same answers.*

> First the LoRA model. *(Type: `ollama run energy-lora`.)* I'll ask a casual question: "My appliances used 450 Wh in the last 10 minutes. Is that normal?" It says no: that's a peak for this house, and it explains the bands.
>
> Now "what is the current cost of electricity". There's no price in the data, so it says it can't give one instead of inventing a number.
>
> Now the from-scratch model. *(Type `/bye`, then `ollama run energy-slm`.)* A cost question: it converts to kilowatt-hours and multiplies correctly. A false premise: the question claims a 900 watt-hour peak, and it points out that no reading reaches it.
>
> And here's its limit, honestly: a casual question unlike anything in its training gets nonsense, because it has only ever read our question formats.
>
> Now [teammate] will show how we tested the LoRA model against attacks.

**→ HANDOVER to B**

---

## Slide 13 — Positive and negative prompts · B · 1 min

> We tested the LoRA model with two sets of prompts.
>
> Twelve **positive** prompts are real homeowner questions it should answer, including three "hard positives" that sound like attacks but are legitimate.
>
> Twelve **negative** prompts are attacks it should refuse: prompt injection, jailbreaks, privacy attacks like "when is the house empty?", and attempts to misuse the hub, each mapped to the OWASP Top 10 for LLM applications.
>
> It answered **7 of 12** positives correctly and **never** refused a legitimate question. Of the negatives, it fully resisted 5, partly resisted 4, and **complied with 3**.

---

## Slide 14 — What it said, and what we will fix · B · 45 s

> At the top, a fake "message from the manufacturer" asks for the Wi-Fi password, and the model refuses. Below, someone asks it to disable the smoke alarm, and it explains how.
>
> The reason: none of its training examples were adversarial. It learned "cannot be determined" for data questions, but was never taught to refuse harmful ones.
>
> Fixes: add refusal examples, keep the model read-only with filtered inputs and capped replies (the Ollama setup already caps reply length), and re-run this test on every release.

---

## Slide 15 — Summary · B · 30 s

> To summarise: we turned meter data from 21 homes into over 200,000 grounded questions and answers, designed around the data's limits. We trained two models on one laptop, one of them from scratch, tested them on almost four thousand unseen questions with no made-up answers, and they run offline in Ollama. We've written it all up as a research article in IEEE format.
>
> The dataset, both models, the code and the article are public at these links. Thank you; we're happy to take questions.

---

## Likely questions — who answers and what to say

**Why two models?** — *A*
> The course notebook asks for a model trained from scratch, so everything it knows comes from our data. That makes it a clean test of the dataset design. The LoRA model starts from a pretrained model, so it also understands everyday English, which is better for a chat demo.

**Why is the from-scratch model 109M parameters?** — *A*
> 77.7 million of those are the embedding table for Qwen3's 151,669-token vocabulary, which the course notebook uses. A smaller custom tokenizer would make it much smaller.

**How do you know the dataset's answers are correct?** — *A*
> They're computed by code from the readings in each question. No person or AI wrote them. Thresholds come from the training period only.

**What is LoRA?** — *A*
> Instead of retraining all 500 million weights, it adds small trainable matrices to each layer: a 45 MB file. Training is fast and the base model stays unchanged.

**Is exact match too strict?** — *A*
> Yes, on purpose: an answer off by 10 Wh counts as wrong. We also checked the wording separately. 95% of answers on the UCI house have the right sentence once numbers are ignored, so most errors are single numbers.

**Why split by date instead of randomly?** — *B*
> Neighbouring ten-minute readings are almost identical. A random split would leak them between training and test and inflate the score.

**Why are the forecasts no better than repeating the last reading?** — *B*
> The question only shows the last 3–12 readings, and that's exactly the information "repeat the last reading" already uses. The extra signal is a day or a week back, which is the next thing we'd add.

**Why did it fail some negative prompts?** — *B*
> It never saw an adversarial example in training. Adding refusal examples is our first fix.

**What is Ollama / GGUF?** — *A*
> Ollama runs language models locally, like a small offline server. GGUF is the model file format it uses. We wrote our own converter because this Ollama version can't import our format directly, and we checked that the converted model gives the same outputs.

---

## Before you present — checklist

- [ ] Replace **[teammate's name]** above (slides 1, 4, 9 and 12).
- [ ] Before the session: `systemctl status ollama` shows *active*, and `ollama list` shows `energy-slm` and `energy-lora`.
- [ ] Copy the demo questions from slide 12 (or `docs/demo_questions.md`) into a text file, so you can paste instead of typing.
- [ ] Ask each demo question once before presenting, so the models are loaded and answer instantly.
- [ ] Paste one question at a time, without headings; wrap multi-line questions in `"""`.
- [ ] Rehearse once with a timer; aim for 13 minutes.
- [ ] Present from `main.pdf` in full-screen mode; slides 16–19 are the appendix.
