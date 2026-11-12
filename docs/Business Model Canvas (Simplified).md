# Business Model Canvas — Simplified (Sheet 9)

Test → Pitch: who would realistically deploy, or benefit from, an SLM trained with the redesigned architecture.

Note: the product is a small forecasting model that runs on the home's energy hub. It reads the last hour of 10-minute readings plus the same time yesterday and last week, forecasts appliance use 10 minutes to 24 hours ahead, flags likely peaks, and answers "Cannot be determined" when the data can't answer. Each box below is tagged with the Sheet 1 problem (#1–#5) or Sheet 2 persona detail it comes from. Where the canvas asks for revenue, this sheet follows the template's tip and also measures impact in kWh shifted, peaks avoided and adviser hours saved.

## Key Partners

- **Smart-home hub and energy-monitor makers**, who build the model into their devices. The model is useless without a device in the home.
- **Distribution grid operators and smart-meter programmes** (for example Fluvius's digital meter roll-out in Flanders). They provide the 10-minute or 15-minute readings, and they gain when household peaks flatten.
- **Energy advisers and local energy agencies**, who turn forecasts into advice for households like Sofie's.
- **Open-weight model providers** (small instruction-tuned models such as Qwen2.5-0.5B or SmolLM2), which give a base model that already understands language.
- **Dataset and research community**: the UCI Appliances Energy Prediction dataset (Candanedo et al. 2017) and the NILM, occupancy and forecasting papers the training tasks are based on.

## Key Activities

- Turn raw meter readings into grounded question-and-answer training data: chronological splits, answers computed from the readings, 50/50 yes/no questions, and trap questions (#1, #3, #4).
- Fine-tune a small base model with LoRA on a single consumer GPU, then compress it (4-bit) so it fits on the hub.
- Adapt the model to each new home from its first weeks of data, on the device (#5).
- Measure it against simple baselines ("next = last", gradient-boosted trees) for forecast error, peak recall and the rate of correct "cannot be determined" answers.
- Retrain each season. The training data only covers winter and spring (#5).

## Value Proposition

**For the homeowner (Sofie):**
- **"Will tonight be a bad one?" answered by the afternoon**, with the expected peak time, so laundry and ironing can move out of it (#2).
- **Peaks, not averages.** The model is trained to catch the rare spikes, where the top 10% of readings hold 38% of the energy (#2).
- **Honest answers.** It says "Cannot be determined" for occupancy, single appliances, faults and tariffs instead of inventing a figure (#4).
- **Correct units.** "60 Wh per 10 min = 360 W", not a figure that reads like kW (#3).
- **Private by design.** It runs on the hub, so the family's daily routine never leaves the house (#5).

**For the adviser:** monthly reviews that start from forecast-vs-actual and the household's real peak pattern, instead of screenshots.

## Customer Relationships

- **Self-service in the app**, with plain-language forecasts and peak alerts before the evening.
- **Trust built through honesty.** Every forecast is labelled an estimate, refusals explain what the data can't show, and the app shows how yesterday's forecast compared with what actually happened.
- **Assisted reviews** once a month with the energy adviser.
- **Learns the home over time**: after a few weeks of local adaptation, the model reflects this household's routine, not a generic one.

## Customer Segments

- **Primary:** homeowners in low-energy or smart-metered homes who pay the bill and decide when big appliances run. Sofie, from Sheet 2, is the representative user.
- **Households on peak-based or time-of-use tariffs**, where a missed peak costs money directly (for example Flanders' capacity tariff, which is based on the household's monthly peak).
- **Energy advisers and agencies** advising many households.
- **Hub and energy-monitor makers**, who want an offline "smart" feature without paying for cloud servers.
- **Secondary:** grid operators and energy suppliers, who benefit from flatter household peaks across many homes.

## Key Resources

- **The training-data generator** (`energydata_to_sml_jsonl.ipynb`) and the ~25,000 grounded question-answer pairs it produces.
- **The fine-tuned, compressed SLM**, plus the redesigned context and attention choices from Sheet 8.
- **Meter and sensor data**: 10-minute appliance readings, room temperature and humidity, and local weather.
- **Hardware**: one consumer GPU for training (an 8 GB RTX-class laptop GPU is enough), and the hub's CPU for running the model.
- **Domain knowledge** from the project's literature corpus on NILM, fault detection, occupancy detection and forecasting.

## Channels

- **Pre-installed on energy hubs**, delivered through the hub maker's firmware or app updates.
- **The hub's mobile and tablet app**: morning and evening forecasts, and peak alerts.
- **Energy advisers and local energy agencies**, who recommend it during home energy audits.
- **Grid-operator and supplier programmes** that bundle it with smart-meter or peak-tariff schemes.
- **Open release** of the model and the data generator for researchers and makers.

## Cost Structure

- **Training:** a few GPU-hours on one consumer GPU per fine-tuning run. The data generator uses only the Python standard library.
- **Base model:** free open-weight models, so no licence fee. Check each model's licence before commercial use.
- **On-device running:** no cloud servers and no per-query API fees. The model runs on hub hardware the household already owns.
- **Ongoing:** seasonal retraining (summer data is missing), per-home adaptation, testing against baselines, and app integration.
- **Data access:** agreements with grid operators or hub makers for real household readings beyond the single UCI house.

## Revenue Streams / Impact Streams

**Revenue (if commercialised):**
- Licensing per device to hub and energy-monitor makers.
- A premium "peak alerts" feature in the hub app.
- Bundling in supplier or grid-operator tariff programmes that reward lower peaks.

**Impact (how success is measured):**
- **kWh shifted out of peak hours** per household per month, and how much the monthly peak drops.
- **Peak recall**: the share of real evening spikes the model warned about in time.
- **Honesty rate**: the share of questions about occupancy, single appliances, faults or tariffs answered with "Cannot be determined" instead of an invented figure.
- **Adviser hours saved** per household review.
- **Privacy**: the amount of household data uploaded to the cloud, which is zero because the model runs on the hub.
