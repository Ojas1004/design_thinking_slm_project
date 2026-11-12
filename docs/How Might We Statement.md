# How Might We Statement (Sheet 3)

**Project:** Applying Design Thinking to Build a Small Language Model on an Energy Prediction Dataset

Define: turn empathy into a sharp design question that points at a Transformer sub-component.

## Point-of-View synthesis

**Pieter, the ML engineer who maintains the hub's forecasting SLM** (Sheet 2), **needs** the model to warn homeowners of peaks that the last two hours give no hint of, **because** the warning is usually there, just a day or a week back. 13% of peaks start after a quiet two hours. For 64% of those, a high reading sat within 30 minutes of the same time yesterday or last week (35% at a random time). But a raw week of readings is about 610,000 characters, so simply adding history cannot fit on a hub.

## Draft How Might We statements

**HMW 1 — context window and attention** ✅ **(our team commits to this one)**
How might we let a hub-sized model reach "same time yesterday" and "same time last week" without reading every hour in between?

**HMW 2 — tokenisation and positional encoding**
How might we encode 10-minute readings so that numbers cost few tokens and each one carries its time of day and weekday, not just its position in the prompt?

**HMW 3 — training objective**
How might we reward the model for catching rare peaks and for answering "cannot be determined", instead of for guessing the 40–60 Wh average?

## Why HMW 1

- It answers Sheet 1's first Transformer limit (fixed context window) and Sheet 2's real ticket: the 850 Wh peak on 26 May, which yesterday's 16:10 jump foretold.
- The literature supports it: most low-voltage forecasting models use lagged load and calendar inputs (LV forecasting review). An attention model on building energy learned a one-day period by itself (Attention Boosted Autoencoder).
- It matches what Sheet 9 promises to customers.

## Checklist for a good HMW (applied to HMW 1)

| Checklist item | HMW 1 |
|---|---|
| Neither too broad nor too narrow | One structure of this data (daily and weekly lags), not "better forecasting" and not "add a head". |
| Names the user's real friction | Peaks the model cannot see coming, which become support tickets and lost homeowner trust. |
| Room for several solutions | Sparse attention over lag positions, calendar-aware positional encoding, retrieved lag tokens, or compressed daily summaries. |
| Answerable by changing a block | Yes: attention pattern, positional encoding or context construction. |

HMW 2 and HMW 3 go forward to Sheet 4 as supporting ideas.
