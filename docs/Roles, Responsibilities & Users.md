# Roles, Responsibilities & Users

**Project:** Applying Design Thinking to Build a Small Language Model on an Energy Prediction Dataset

**The product, in one line:** a small language model that runs offline on a home energy hub, forecasts the house's appliance electricity use 10 minutes to 24 hours ahead, warns about evening peaks, and answers "Cannot be determined" when the meter data can't answer a question.

---

## 1. Which companies would realistically deploy it

| Company type | Why they would use it | How likely |
|---|---|---|
| **Smart-home hub and home energy management system (HEMS) makers** | They already sell the hub that sits in the home and reads the meter. An offline assistant is a selling feature with no cloud server costs. | **Most likely.** This is the natural owner of the product. |
| **Solar, battery and heat-pump installers** | Their customers need to know when the house will draw the most power, so they can time battery discharge and appliance use. | Likely, usually by licensing the model from a hub maker. |
| **Energy suppliers and utilities** | Customers on time-of-use or peak-based tariffs (for example Flanders' capacity tariff) pay more for peaks. An app feature that flattens peaks reduces complaints and grid strain. | Possible, as a feature inside their customer app. |
| **Energy advisory agencies** | Advisers can start each household review from forecast-vs-actual and the real peak pattern, instead of screenshots. | Likely as **users**, not builders. |

**Realistic owner:** a hub or HEMS maker builds and ships it. Utilities, installers and advisers use it or bundle it.

---

## 2. Who builds and runs it (inside the company)

### Product & design

| Role | Responsibilities |
|---|---|
| **Product Manager (Energy / Smart Home)** | Owns the problem statement and the roadmap. Decides which questions the assistant must answer (peak alerts, comparisons, units) and which it must refuse (occupancy, single appliances, faults, tariffs it doesn't know). Defines success: peak recall, honest-refusal rate, kWh shifted out of peaks. Prioritises the known gaps: day-ahead forecasts, summer data, more houses. |
| **UX Designer / UX Writer** | Designs how forecasts, peak alerts and "Cannot be determined" answers look in the app. Writes plain-language wording such as "590 Wh in 10 minutes ≈ 3.5 kW", and labels every forecast as an estimate. Runs usability tests with real households (the Sheet 7 test plan). |

### Data & machine learning

| Role | Responsibilities |
|---|---|
| **Data Engineer** | Builds the pipeline from meters and sensors to clean 10-minute data: parsing, unit checks (Wh per 10 minutes, not kW), gap detection, chronological train/val/test splits. Brings in data from new houses and seasons beyond the single UCI house. Owns the raw data and the data dictionary (for example, T6 is outdoors and rv1/rv2 are noise). |
| **ML Engineer (SLM / Fine-tuning)** | Owns the training-data generator (`energydata_to_sml_jsonl.ipynb`) and keeps its rules: answers computed from the prompt, 50/50 yes/no questions, no leakage between splits. Picks the base model (for example Qwen2.5-0.5B or SmolLM2), fine-tunes it with LoRA, and compresses it to 4-bit so it fits on the hub. Adds the yesterday/last-week context and day-ahead tasks. |
| **Data Scientist (Forecasting & Evaluation)** | Builds the baselines the SLM must beat: "next = last value" and gradient-boosted trees. Measures forecast error in Wh, peak recall, calibration of "estimate" ranges and the hallucination rate on trap questions. Reports where the SLM wins and where it doesn't. |
| **MLOps Engineer** | Versions models and training data, automates seasonal retraining (the model has never seen summer), runs staged rollouts to hubs, and watches for drift, such as forecasts getting worse after a household's routine changes. |

### Device & app

| Role | Responsibilities |
|---|---|
| **Embedded / Edge Software Engineer** | Runs the compressed model on the hub's CPU within its memory and speed limits (for example llama.cpp or ONNX Runtime). Handles on-device adaptation to each home and delivers model updates through firmware. Makes sure no household data leaves the device. |
| **Mobile App Developer** | Builds the chat, forecast and peak-alert screens in the hub app, and the monthly forecast-vs-actual summary for advisers. Connects the app to the on-device model over the home network. |

### Quality, trust & domain

| Role | Responsibilities |
|---|---|
| **Energy Domain Expert / Energy Analyst** | Checks that answers make physical sense: units, power vs energy, typical appliance loads, tariff rules. Reviews the trap questions and refusal cases. Translates tariff schemes, such as peak-based billing, into product requirements. |
| **QA / Model Evaluation Engineer** | Maintains the held-out test set and the trap and "cannot be determined" suites. Runs regression tests before every model release, so a new model never starts inventing appliance figures or mixing up Wh and kW. |
| **Privacy & Compliance Officer (Data Protection Officer)** | 10-minute household energy data reveals when people are home or asleep, so in the EU it counts as personal data under GDPR. Approves what is stored, for how long, and what (if anything) leaves the hub. Signs off on consent for any opt-in data sharing used for retraining. |

### Customer-facing

| Role | Responsibilities |
|---|---|
| **Customer Support Specialist** | Answers "why did it say this?" questions. Escalates wrong forecasts and wrong refusals to the ML team as labelled examples for the next training round. |
| **Partnerships / Business Development Manager** | Signs licensing and bundling deals with installers, utilities and grid operators, and secures access to meter data from more homes. |

**In a startup**, these roles collapse into a team of 4–6: one PM/UX person, one or two ML engineers who also do data engineering and evaluation, one embedded/app engineer, and a part-time energy expert and privacy adviser.

---

## 3. Who uses its output

| User | What they use it for | What they need from it |
|---|---|---|
| **Homeowner / bill payer** (primary; Sofie in Sheet 2) | Asks "Will tonight be a bad one?", gets afternoon peak alerts, moves laundry and ironing out of the peak, compares today with yesterday or last week. | Plain answers, correct units, honest "Cannot be determined", and data that stays at home. |
| **Other household members** | Quick checks from the kitchen tablet before switching on the oven, dryer or iron. | One-line answers, no graphs. |
| **Energy Adviser** | Monthly or annual household reviews, starting from forecast-vs-actual and the household's real peak pattern. | A summary they can trust, with forecasts clearly marked as estimates. |
| **Installer (solar, battery, heat pump)** | Sizes batteries and sets charge/discharge times around the forecast evening peak. | Reliable peak timing and size, for each household. |
| **Utility / supplier customer team** | Offers the feature to customers on peak-based or time-of-use tariffs, and handles fewer bill complaints. | Evidence that it reduces customer peaks. |
| **Grid operator / demand-response planner** (secondary, opt-in only) | Uses anonymised, aggregated forecasts from many homes to anticipate neighbourhood peaks. | Aggregated numbers only. Individual households stay private. |
| **Customer Support Specialist** | Explains the assistant's answers to customers and logs mistakes. | The question, the answer, and what the data actually showed. |

---

## 4. Who owns which part of this project today

| Project file or part | Owner role |
|---|---|
| `problem_framing_answers.md`, Sheets 1–11 | Product Manager, UX Designer |
| `energydata_complete.csv` (cleaning, units, splits) | Data Engineer |
| `energydata_to_sml_jsonl.ipynb` and `energydata_sml_*.jsonl` | ML Engineer |
| Baselines and the test-set scorecard (not built yet) | Data Scientist, QA / Evaluation Engineer |
| LoRA fine-tuning and 4-bit compression (not built yet) | ML Engineer, Embedded Engineer |
| `0_gpu_check.ipynb` and the training environment | ML Engineer, MLOps Engineer |
| Literature corpus (NILM, faults, occupancy, forecasting) | Data Scientist, Energy Domain Expert |
| Privacy design (on-device only, no cloud upload) | Privacy & Compliance Officer, Embedded Engineer |
