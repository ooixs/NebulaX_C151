# ACV refrigerant-leak ranking

## Summary

The Air-Conditioning and Ventilation (ACV) task ranks the cars in a train from most to least
likely to have a refrigerant leak. The final method compares each car's cabin temperature with
the other cars operating at the same time. When refrigeration-pressure telemetry is available,
it also uses the pressure deficit expected from refrigerant undercharge.

Leave-one-case-out evaluation over the six labelled cases produced the official rank-decay score
**1.000**. This is a useful result, but the dataset is very small and the pressure rule is tested
on only one rich-format case.

## Problem and data

Each case is an Excel workbook with timestamped readings for the cars in one train. Most cases
contain eight ACV parameters per car, including indoor temperature, setpoint, operating mode,
outdoor temperature, load status, and data-validity status. One training case contains more than
60 parameters per car, including refrigeration pressures. The exact column set is not fixed.

There are six labelled training cases and one unlabelled test case. The output for each case is a
`|`-separated list of every car identifier, ordered from most to least likely faulty. The
official rank-decay score decreases linearly as the true faulty car moves down the list.

## Loading and assumptions

Columns are parsed with the pattern `Car <NN> - <parameter>`. Parameter names are mapped to roles
by keywords rather than a fixed list, allowing the common and rich formats to use the same
pipeline. Zero-degree readings and rows marked `Invalid` are masked instead of treated as fault
signals.

The method assumes that cars in the same train experience broadly comparable ambient and service
conditions at each timestamp. This makes the other cars useful contemporaneous references. The
method does not assume that absolute cabin temperatures are comparable across separate case
files.

## Validation design

The six cases are the independent units. Leave-one-case-out validation fits any learned response
model on five complete cases and ranks the cars in the sixth. Rows or cars are never randomly
split across training and validation.

The final scorer has fixed physical weights, so its six-case score does not involve refitting
those weights in each fold. This avoids the optimistic result obtained when a multi-component
weight vector is chosen and scored on all six cases. However, the final pressure rule was still
developed after examining these cases. The 1.000 result is therefore not an independent test
score.

## Exploratory findings

### Peer-relative temperature

For each timestamp, the pipeline compares a car's indoor temperature with the median of cars in
the same cooling mode. The deviations are aggregated over the case. In the five common-format
training cases, the faulty car is 0.46-1.31 degrees Celsius warmer than its peers on average,
while healthy cars remain within approximately 0.35 degrees of the peer reference.

This simple signal places the faulty car first in five cases. In the rich-format case, the true
faulty car is second because another car has a warmer cabin while operating at Full Cooling for
most of the recording.

### Pressure telemetry

The rich-format case clarifies that ambiguity. The labelled faulty car's high-side pressure is
about 250 kPa below its peers, while the warmer healthy car's pressure is about 120 kPa above its
peers. A peer-relative discharge-pressure deficit therefore provides a direct signal of
undercharge that cabin temperature alone misses.

## Methods tested

| Method | Leave-one-case-out rank-decay | Decision |
|---|---:|---|
| Mean peer-temperature deviation | 0.979 | Baseline; true car ranked second once |
| Six-component weighted score, weights chosen on the other five cases | 0.958 | Rejected |
| Same weighted score chosen and scored on all six cases | 1.000 | Rejected as optimistic |
| Setpoint-adjusted or trimmed peer deviations | 0.979 | Tied baseline |
| Borda aggregation of several indicators | 0.958 | Rejected |
| High-load and cooling-duty variants | 0.792-0.813 | Rejected |
| Peer-temperature deviation plus fixed pressure deficit | **1.000** | Selected |

The six-component score included setpoint error, persistence, positive-deviation fraction, a
healthy-response residual, and pressure information. Its lower nested result shows that tuning
many weights on six cases is unstable. Simpler temperature variants did not solve the rich-case
ambiguity.

The selected pressure term has a fixed weight and is active only when the relevant columns
exist. Weights from 0.5 to 2.0 gave the same ranking in the rich-format case. On the common
eight-parameter format, the term is absent and the method reduces to peer-temperature ranking.

## Final product method

The active artifact is from run `2026-09-18-autoresearch`:

1. Read each file's own headers and construct one time series per car.
2. Mask readings marked invalid.
3. Restrict temperature comparisons to relevant cooling operation.
4. Compute each car's deviation from its contemporaneous peers.
5. Add a discharge-pressure-deficit term when compatible pressure telemetry is present.
6. Rank every observed car by the combined anomaly score.

The current test ranking is `01|04|03|07|08|06|02|05`. Car 01 leads the available temperature
indicators, but its margin is smaller than in any common-format training case. The ranking is a
model output, not a confirmed leak diagnosis.

## Why this fits maintenance work

The output is deliberately a ranking rather than a binary fleet-wide alarm. It tells maintenance
staff where to begin a refrigerant inspection while preserving the remaining cars as a follow-up
order. Peer comparison also reduces dependence on a single absolute-temperature threshold when
weather, occupancy, and setpoints change.

## Limitations

- Six labelled cases are not enough to establish performance across the full range of trains,
  weather, occupancy, and ACV configurations.
- Only one case contains useful pressure telemetry, so the pressure rule needs more independent
  validation.
- The common-format test signal is weaker than the temperature separation seen in training.
- The ranking localises a likely car. It does not confirm a leak or estimate refrigerant loss.
- Missing data can reduce the number of comparable peers at a timestamp.

## Reproduction pointers

- Pipeline and feature roles: `ACV/code/pipeline.py`
- Training and leave-one-case-out evaluation: `ACV/code/train.py`
- Prediction: `ACV/code/predict.py`
- Exploratory feature table: `ACV/notebooks/explore.py`
- Alternative scorers: `ACV/notebooks/autoresearch.py`
- Official metric: `common/metrics.py`

## References

- ACV Subsystem Info Kit, `docs/references/ACV/ACV_Subsystem_Info_Kit.md`.
- Du, Domanski and Payne (2015), operating-condition effects in vapour-compression fault data,
  [NIST](https://www.nist.gov/publications/effect-common-faults-performance-different-vapor-compression-systems).
- Mattera et al. (2018), residual-based HVAC fault detection,
  [Sensors](https://www.mdpi.com/1424-8220/18/11/3931).
