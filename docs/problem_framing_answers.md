# Problem Spotter Sheet (Sheet 1)

## Chosen dataset & source (link / citation)

UCI Appliances Energy Prediction (archive.ics.uci.edu/dataset/374); Candanedo, Feldheim & Deramaix, Energy and Buildings 140 (2017). 19,735 ten-minute readings from one house, Jan–May 2016.

## Domain / specific area of focus

Short-horizon (10 min to 24 h) forecasting of appliance electricity use in one low-energy Belgian house, from its 10-minute sub-meter, nine room temperature/humidity sensors and local weather, for a small model running on the home's energy hub.

## Who deals with this data today, and how are they underserved?

*(The form has no blank under this heading, so the users are named inside each "why it matters" cell. The users are: the homeowner reading the hub's app, and the energy adviser who recommends changes to them.)*

| # | PROBLEM / GAP OBSERVED | WHY IT MATTERS TO THE USER |
|---|---|---|
| 1 | Tokenisation: readings are numbers, not words. One row is about 550 characters and one day about 79,000, and BPE splits each number into arbitrary fragments. | The homeowner gets wrong answers to "is today higher than yesterday?", and one day of history already overflows a hub-sized model. |
| 2 | Regression to the mean: the target is heavy-tailed (median 60 Wh, max 1,080 Wh) and the top 10% of readings hold 38% of all energy. A generic objective learns the typical 40–60 Wh. | Peaks drive the bill and peak tariffs, so a forecast that misses them is useless to the homeowner and the adviser. |
| 3 | No labels for occupancy, individual appliances, tariffs or faults, but next-token training rewards a fluent answer over "cannot be determined". | A homeowner acting on an invented "your dishwasher used X" makes the wrong change and stops trusting the tool. |
| 4 | Noisy and misleading channels: rv1/rv2 are random, T6 is the outdoor sensor (r = 0.97 with T_out), nine room temperatures move together, and values are Wh per 10 min (60 means 360 W), not kW. | Attention spread over noise yields false explanations ("bathroom humidity drives usage"), and unit slips give the user power figures that are 6× off. |
| 5 | One house, 4.5 months, winter to spring only, and the trace exposes daily routines. Pretrained models don't know this household, fine-tuning on 19.7k rows overfits, and cloud models need the data uploaded. | The homeowner needs a model that adapts to their home from little data, runs locally to keep routines private, and admits it has never seen a summer. |

## Where specifically does a vanilla Transformer struggle with this data's structure?

**Fixed context window:** beyond the last hour (lag-1 autocorrelation 0.75, lag-6 0.32), the next useful signal sits a day (144 steps) and a week (1,008 steps) back, with almost nothing in between (lag-36 ≈ 0). Serialised as text, one day is about 79,000 characters, so a 2k–8k-token window holds only a few hours: it keeps the uninformative middle and never sees "same time yesterday".

**Token-order positional encoding:** the data is a 2-D grid (time × 28 channels on a strict 10-minute clock), but token positions flatten it into one line. Numbers differ in length ("60" vs "19.890000000000001"), so the token distance between consecutive timestamps varies; the 28 values of one row get different positions although they share one instant; and nothing encodes hour-of-day or weekday, which drive the 6 pm (190 Wh) and Monday peaks.

**Dense quadratic attention:** cost grows with the square of (time steps × 28 channels × tokens per value), so a week of history is unaffordable on a hub. Most token pairs carry no signal: rv1/rv2 are noise, T6 duplicates T_out, the room temperatures are near-collinear, and hourly weather is interpolated across six rows. Softmax over hundreds of near-identical 50–60 Wh tokens dilutes the rare 3–4-step spikes that hold 38% of the energy.
