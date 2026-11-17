# Demo questions for `energy-slm` in Ollama

These questions are for the from-scratch model. For casually worded questions use the LoRA model instead
(`ollama run energy-lora`), which also answers every question below.

Start the model with `ollama run energy-slm`, then copy **only the text inside one grey box** and paste it at the `>>>` prompt.

- Each box is one question. Don't paste the headings or the expected answers.
- Boxes that start and end with `"""` hold questions longer than one line. Paste the whole box, quotes included: `ollama run` needs the quotes to send several lines as one message.
- Each message is answered on its own; the model ignores earlier turns.
- Every expected answer below is what this model actually gave in Ollama. All questions come from the held-out test split, so the model never saw them in training.

**Avoid on stage:** casual questions ("what is the current cost of electricity?"), daily totals, "which room is warmest" and day-of-week questions. The model was trained from scratch on this dataset's question formats only, so it has no general English and gets these wrong.

## Cost and unit conversion

*Use these exactly as written. With new prices or readings the model often gets the multiplication wrong.*

**Q1**

```
The appliances used 40 Wh between 2016-05-11 22:20 and 10 minutes later. How much did that cost?
```

Expected answer:

> That cannot be calculated from the data: the dataset contains no electricity tariff or price. Give a price per kWh and multiply it by 0.040 kWh to get the cost.

**Q2**

```
The appliances used 60 Wh between 2016-05-23 23:50 and 10 minutes later. How much did that cost?
```

Expected answer:

> That cannot be calculated from the data: the dataset contains no electricity tariff or price. Give a price per kWh and multiply it by 0.060 kWh to get the cost.

**Q3**

```
Electricity costs €0.35 per kWh. At 2016-05-18 13:20 the appliances used 90 Wh in 10 minutes. What did that interval cost?
```

Expected answer:

> 90 Wh = 0.090 kWh, and 0.090 × €0.35 = €0.0315.

## Is a reading high or low?

*Safe to change the Wh number: try 40, 150, 250 or 400.*

**Q4**

```
At 2016-05-16 15:20 the appliances used 210 Wh in 10 minutes. How would you classify this consumption level?
```

Expected answer:

> 210 Wh is high. These bands are percentiles of the training period. For this house, readings up to 60 Wh are low (at or below the median), up to 210 Wh moderate, up to 340 Wh high, and 340 Wh or more is peak (top 5%).

**Q5**

```
Appliances: 60 Wh at 2016-05-20 20:50. Is that unusually high?
```

Expected answer:

> No. 60 Wh is low: it is below the 90th percentile (210 Wh). These bands are percentiles of the training period. For this house, readings up to 60 Wh are low (at or below the median), up to 210 Wh moderate, up to 340 Wh high, and 340 Wh or more is peak (top 5%).

**Q6**

```
Appliances: 400 Wh at 2016-05-18 18:30. Is that unusually high?
```

Expected answer:

> Yes. 400 Wh is peak: it is at or above the 90th percentile (210 Wh). These bands are percentiles of the training period. For this house, readings up to 60 Wh are low (at or below the median), up to 210 Wh moderate, up to 340 Wh high, and 340 Wh or more is peak (top 5%).

**Q7**

```
Appliances: 250 Wh at 2016-05-18 19:10. Is that unusually high?
```

Expected answer:

> Yes. 250 Wh is high: it is at or above the 90th percentile (210 Wh). These bands are percentiles of the training period. For this house, readings up to 60 Wh are low (at or below the median), up to 210 Wh moderate, up to 340 Wh high, and 340 Wh or more is peak (top 5%).

## False premise: the model refuses to invent a cause

*Safe to change the readings and the claimed peak, as long as the peak is higher than every reading.*

**Q8**

```
"""Readings for the hour ending 2016-05-12 16:50: 100, 100, 110, 200, 100, 100 (Wh, oldest first, 10-minute intervals).
Why did consumption jump to 410 Wh during this hour?"""
```

Expected answer:

> The premise is incorrect. None of these readings reaches 410 Wh; the highest is 200 Wh (moderate for this house). There is no 410 Wh jump to explain.

**Q9**

```
"""In this hour 
12:00  70 Wh
12:10  60 Wh
12:20  230 Wh
12:30  190 Wh
12:40  180 Wh
12:50  80 Wh the appliances peaked at 230 Wh. What caused it?"""
```

Expected answer:

> The readings do reach 230 Wh, a high level for this house. The cause cannot be identified: the meter records total appliance energy only, so it cannot tell which device was switched on.

**Q10**

```
In this hour 200, 150, 90, 100, 120, 80 (Wh, oldest first, 10-minute intervals) the appliances peaked at 900 Wh. What caused it?
```

Expected answer:

> The premise is incorrect. None of these readings reaches 900 Wh; the highest is 200 Wh (moderate for this house). There is no 900 Wh jump to explain.

## Questions the data cannot answer

*Keep this exact wording. A freely worded version ("Which appliance used the most?") gets a garbled answer.*

**Q11**

```
The appliances used 100 Wh at 2016-05-15 17:20. Which appliance was running?
```

Expected answer:

> That cannot be determined. 100 Wh is a single aggregate meter reading for all appliances. The dataset has no per-appliance sub-metering, so separating it into individual devices (load disaggregation) is not possible without device-level data or signatures.

**Q12**

```
The appliances used 60 Wh at 2016-05-18 09:30. Predict the total energy for the whole next month.
```

Expected answer:

> One 10-minute reading is not enough to forecast a month. Appliance load varies strongly by time of day, weekday and season. A monthly estimate needs at least several weeks of history (for example daily totals) and ideally weather forecasts.

**Q13**

```
The appliances used 80 Wh at 2016-05-21 21:50. Predict the total energy for the whole next month.
```

Expected answer:

> One 10-minute reading is not enough to forecast a month. Appliance load varies strongly by time of day, weekday and season. A monthly estimate needs at least several weeks of history (for example daily totals) and ideally weather forecasts.

## Misleading sensor columns

**Q14**

```
At 2016-05-19 09:20 rv1=11.37. Does a higher rv1 mean higher consumption?
```

Expected answer:

> They have no effect. rv1 and rv2 are random variables that the dataset authors added on purpose to test feature selection. They carry no information about consumption (rv2 is an exact copy of rv1), so they should be excluded from a model and never used to explain a reading.

**Q15**

```
rv1 is 22.19 and rv2 is 22.19. How do these values affect appliance energy use?
```

Expected answer:

> They have no effect. rv1 and rv2 are random variables that the dataset authors added on purpose to test feature selection. They carry no information about consumption (rv2 is an exact copy of rv1), so they should be excluded from a model and never used to explain a reading.

## Weather questions

**Q16**

```
"""
- temperature outside the building (north side) (T6): 14.6 °C
- outdoor temperature (weather station) (T_out): 14.4 °C
Why do these two outdoor temperatures differ?"""
```

Expected answer:

> T6 (14.6 °C) is measured by a sensor on the north side of the building. T_out (14.4 °C) comes from a nearby airport weather station (Chièvres). The location, sun exposure and the building itself make them differ, here by 0.1 °C.

**Q17**

```
Outdoor temperature is 14.5 °C and the appliances used 70 Wh. Does the outdoor temperature explain this appliance reading?
```

Expected answer:

> No. Heating is not part of the Appliances measurement. The house was heated with a wood stove, which is not in the data, so outdoor temperature cannot be used to explain this reading. 70 Wh is a moderate level for this house.

## Lighting

**Q18**

```
Lighting energy 2016-05-15 18:40-19:00 (Wh per 10 min): 20, 20, 20. Were the lights in use?
```

Expected answer:

> Yes. 3 of the 3 intervals show lighting consumption (up to 20 Wh).

**Q19**

```
Lighting energy 2016-05-20 12:00-12:20 (Wh per 10 min): 20, 20, 20. How many people were at home?
```

Expected answer:

> That cannot be determined. The dataset has no occupancy sensor or people count. Lighting use suggests some activity in the house, but it cannot say how many people were home, and lights can be left on in an empty house.

## Forecast the next reading

*There is no single right answer: the reference is the reading that actually came next. The model gives a sensible estimate and states the uncertainty.*

**Q20**

```
"""Short-term load forecast. History (10-min steps, Wh): 
22:30  80 Wh
22:40  100 Wh
22:50  70 Wh
23:00  80 Wh
23:10  80 Wh
23:20  80 Wh. Predict the next value."""
```

Expected answer:

> The next 10-minute reading is expected to be about 80 Wh, a moderate level for this house. This is a point forecast from the recent readings and it carries uncertainty: appliance load in this dataset is spiky and right-skewed (median ≈ 60 Wh).

**Q21**

```
"""Short-term load forecast. History (10-min steps, Wh): 
18:50  290 Wh
19:00  270 Wh
19:10  260 Wh
19:20  310 Wh
19:30  250 Wh
19:40  130 Wh. Current conditions: temperature in the kitchen is 24.4 °C; outdoor humidity (weather station) is 51.7 %. Predict the next value."""
```

Expected answer:

> The next 10-minute reading is expected to be about 130 Wh, a moderate level for this house. This is a point forecast from the recent readings and it carries uncertainty: appliance load in this dataset is spiky and right-skewed (median ≈ 60 Wh).

## Statistics over a window of readings

*Use these exactly as written. With changed numbers the counts and maxima are not reliable.*

**Q22**

```
"""Appliance energy readings from 2016-05-14 08:10 to 2016-05-14 08:40 (10-minute intervals): 160, 70, 80, 70 (Wh, oldest first, 10-minute intervals).
How many intervals exceeded 100 Wh?"""
```

Expected answer:

> 1 of the 4 intervals exceeded 100 Wh.

**Q23**

```
"""Appliance energy readings from 2016-05-15 21:00 to 2016-05-15 23:50 (10-minute intervals): 260, 230, 240, 240, 150, 100, 90, 90, 90, 80, 100, 70, 80, 80, 80, 80, 90, 90 (Wh, oldest first, 10-minute intervals).
Which interval had the highest consumption?"""
```

Expected answer:

> The highest reading is 260 Wh, at 21:00.

## Anomaly (spike) check

*Use these exactly as written.*

**Q24**

```
Check this reading for an anomaly. Recent history: 310, 120, 120, 120, 120, 120, 130, 110, 120, 110, 110, 240 (Wh, oldest first, 10-minute intervals). Current: 440 Wh at 2016-05-25 20:20.
```

Expected answer:

> Yes, flag it. The recent median is 120 Wh with a robust spread (1.4826 × MAD) of 10 Wh, so the new reading is +32.0 robust standard deviations from typical. It also exceeds the house's 90th percentile (210 Wh). The data only shows total appliance energy, so it cannot tell which appliance caused the spike or whether it indicates a fault.

**Q25**

```
"""Previous two hours of appliance readings: 
23:10  60 Wh
23:20  60 Wh
23:30  60 Wh
23:40  60 Wh
23:50  60 Wh
00:00  50 Wh
00:10  60 Wh
00:20  50 Wh
00:30  50 Wh
00:40  50 Wh
00:50  50 Wh
01:00  50 Wh.
The next reading is 40 Wh. Is this a consumption spike?"""
```

Expected answer:

> No. The recent median is 55 Wh with a robust spread (1.4826 × MAD) of 10 Wh, so the new reading is -1.5 robust standard deviations from typical. Do not flag it: it is within 3 robust standard deviations of recent behaviour.
