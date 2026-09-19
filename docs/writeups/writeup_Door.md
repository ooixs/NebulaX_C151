# Door-cycle segmentation and resistance classification

## Summary

The Door subsystem processes one continuous stream of controller and motor measurements. It must
find every opening or closing cycle and label each cycle as `Normal` or `Abnormal resistance`.

We treat segmentation and classification as distinct sub-problems. Segmentation uses the timing
of the continuous stream to find cycle boundaries. Classification then evaluates each cycle on
its own. The files contain no train or door identifier, so the classifier cannot compare a door
with its own history or use other cycles from the same physical door as context. The two stages
are nevertheless evaluated together because the official IoU-weighted F1 score requires both the
boundary and the label to be correct.

The product uses timestamp-gap segmentation followed by a Random Forest and logistic-regression
ensemble. Five-block validation produced an end-to-end IoU-weighted F1 of **1.000** on the
available labelled stream.

## Problem and data

The training stream has 18,036 rows and 17 columns. It contains 110 labelled cycles: 80 Normal
and 30 Abnormal resistance. Measurements arrive every 20 ms while a door is moving. The columns
include motor current, motor voltage, back-EMF, controller commands, limit-switch states, and door
position. The unlabelled test stream has 6,253 rows.

The required output contains one row per detected cycle:

- `start_time`
- `end_time`
- `prediction`, either `Normal` or `Abnormal resistance`

The official metric matches predicted and true intervals of the same class, weights each match by
intersection over union, and combines the soft precision and recall as an F1 score. It therefore
penalises missed cycles, extra cycles, poor boundaries, and wrong labels.

## Validation design

Random row splitting would place samples from the same physical cycle in training and validation.
Instead, cycles are kept intact and divided into five contiguous time blocks. Current references
and classifiers are fitted using the training blocks only. For the selected ensemble, the
classification threshold is also chosen by inner blocked validation rather than from the outer
validation labels.

The full validation path is:

1. Segment the held-out continuous block.
2. Extract one feature vector per predicted cycle.
3. Classify each cycle.
4. Score the complete predicted intervals and labels with the official metric.

This design measures the same combined task required at inference time.

## Segmentation

### Exploratory findings

The timestamp pattern separates the cycles directly. Consecutive rows within a labelled cycle
are 20 ms apart, while the silence between cycles is 10 to 59 seconds. A threshold between these
ranges is therefore stable rather than finely tuned.

### Methods considered

We first implemented a timestamp-gap segmenter. A new cycle starts whenever the gap between two
rows exceeds one second. Motion-state and change-point segmenters were considered as fallbacks,
but further complexity was not justified after the simple rule reproduced all 110 labelled
cycles exactly.

### Result and final method

The one-second gap rule achieved segmentation-only IoU-weighted F1 **1.000**. It preserved the
start-up, travel, and locking phases expected by the reference intervals. Applied to the test
stream, it produced 38 complete cycles, with every input row assigned to a cycle.

The main assumption is that the unlabelled stream was collected with the same silence between
cycles. Its observed gaps support that assumption, but the labels are private and the resulting
boundaries cannot be scored locally.

## Classification

### Exploratory findings

Motor current changes across a door movement. The locking peak is not the clearest fault signal:
peak current, cycle duration, and the controller's opening or closing time do not separate the
training classes.

The useful signal occurs during travel. Mean current over the middle 30-80% of a cycle is
186-195 mA for Normal closing cycles and at least 313 mA for Abnormal closing cycles. For opening
cycles, the corresponding ranges are 203-221 mA and at least 235 mA. This supports comparing
phase-specific current rather than relying on one maximum value.

### Features

Each cycle is described by:

- current, voltage, back-EMF, and position statistics;
- mean current during the middle of travel;
- duration and motion features; and
- current excess relative to a median and interquartile-range Normal profile for the same
  operation, fitted within the training fold.

Absolute current is retained. Normalising every cycle by its own amplitude would remove part of
the resistance signal.

### Model comparison

| Method | Five contiguous blocks | End-to-end IoU-F1 | Outcome |
|---|---:|---:|---|
| Random Forest, 500 balanced trees | 1.000 in every block | 1.000 | Strong baseline |
| LightGBM, class balanced | 1.000 in every block | 1.000 | Tied baseline |
| Four-feature logistic regression | Lower overall | 0.982 | Retained only as a second opinion |
| Random Forest + logistic regression | 1.000 in every block | **1.000** | Selected |

Several models reached the score ceiling, so the final choice was based on margin behaviour and
physical interpretation rather than a higher validation score. The selected model averages the
Random Forest and logistic-regression probabilities. Its threshold is approximately 0.33, the
centre of the interval that made no out-of-fold errors.

### Borderline test cycles

Seven unlabelled test cycles fall between the Normal and Abnormal training current ranges. This is
not validation evidence because their true labels are unavailable. The selected model marks them
Normal for two reasons:

- Their current excess is a short start-up spike, while every training abnormal cycle has
  sustained excess through acceleration and travel.
- Limit-switch timing, which is not used by the classifier, matches the Normal training pattern.

These two signals reduce reliance on one model's extrapolation, but they cannot replace held-out
labels.

## Final product method

The active product artifact is from run `2026-09-18-autoresearch`:

1. Split the stream at timestamp gaps above one second.
2. Infer opening or closing from the movement direction and controller signals.
3. Extract phase-specific electrical and motion features.
4. Compare current with the fold-derived Normal profile for that operation.
5. Apply the Random Forest/logistic-regression ensemble and its fixed threshold.
6. Return the original start and end timestamps with the predicted class.

The current test export contains 38 cycles: 30 Normal and 8 Abnormal resistance. These are model
predictions, not verified test results.

## Limitations

- The training data has no train or door identity, so the model cannot learn a door-specific
  baseline or degradation trend.
- Validation reaches 1.000 on one labelled stream. This does not show how the method behaves on a
  different train, door design, or operating environment.
- The seven borderline test cycles remain uncertain because no test labels are available.
- A changed logger cadence or shorter gap between cycles would require the segmentation rule to
  be checked again.

## Reproduction pointers

- Pipeline: `Door/code/pipeline.py`
- Training and blocked validation: `Door/code/train.py`
- Prediction: `Door/code/predict.py`
- Exploratory segmentation: `Door/notebooks/explore.py`
- Switch-timing check: `Door/notebooks/switch_timing.py`
- Official metric: `common/metrics.py`

## References

- Door Subsystem Info Kit, `docs/references/Door/Door_Subsystem_Info_Kit.md`.
- Wei et al. (2020), phase-wise motor-current analysis for door resistance,
  [Urban Rail Transit](https://link.springer.com/article/10.1007/s40864-020-00133-4).
