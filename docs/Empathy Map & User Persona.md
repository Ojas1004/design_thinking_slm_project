# Empathy Map & User Persona (Sheet 2)

**Project:** Applying Design Thinking to Build a Small Language Model on an Energy Prediction Dataset

## PERSONA CARD

**Name & Role:** Pieter Janssens, 31, ML Engineer at a home-energy-hub maker. He maintains the hub assistant, our SLM, and owns its training data, fine-tuning, testing and seasonal retraining.

**Context of use:** At his desk with one GPU and test hubs. Every day he traces support tickets to the training data, and he tests every release on 4,547 held-out questions.

*Real case:* on Thu 26 May the house was flat (50–80 Wh) until 15:50, then hit **850 Wh at 16:40**. The model saw only the last 2 hours, but **yesterday** jumped at the same time (420 Wh at 16:10).

## Goals

- Beat simple baselines on **peaks**, not just on average error.
- Never invent appliance, occupancy or tariff answers.
- Fit the hub (4-bit, a few hundred MB) and keep data off the cloud.

## Frustrations

- Numbers fill the context window, so prompts hold only 2 hours. (#1)
- Training pulls forecasts to the 40–60 Wh average; the top 10% of readings hold 37% of the energy. (#2)
- Misleading channels: `T6` is outdoors, `rv1`/`rv2` are noise, and Wh vs kW. (#3)
- With no labels for occupancy, appliances or faults, the model wants to guess. (#4)
- One house, January to May only: no summer, and it overfits. (#5)

## Empathy Map

| SAYS | THINKS |
|---|---|
| "It wasn't wrong, it never saw yesterday." | "Average error looks fine; users remember the missed peaks." |
| "No release until it beats 'next = last value'." | "What happens in July?" |

| DOES | FEELS |
|---|---|
| Replays each ticket with the exact prompt the model saw. | Frustrated when a fixable peak is missed. |
| Writes trap and "cannot be determined" questions, and rolls out to 5% of hubs first. | Anxious before releases; relieved that data stays on the hub. |

*Representative persona built from the project's data, not an interview. Tags (#1–#5) refer to Sheet 1 problems.*
