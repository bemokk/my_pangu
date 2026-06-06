# Wind-Wave Baseline and Training Diagnostics Design

## Goal

Make wind-to-wave experiments scientifically comparable before increasing training cost.

## Persistence Baseline

For each initialization time `t0`, use the observed wave field at `t0` as the
prediction for every configured future lead. The baseline contains the same four
channels as the model target:

- `swh`
- `mwp`
- `cos_mwd`
- `sin_mwd`

The dataset returns the persistence forecast normalized with the same target
statistics as the model targets. Baseline loss and physical metrics therefore use
the same masks, denormalization, and metric functions as model evaluation.

Training writes validation and test persistence results to:

`Wind_Wave/outputs/logs/baseline_metrics_by_lead.csv`

## Training Curve

Training continues to write one row per epoch to `train_log.csv`. After training,
the pipeline plots train and validation loss against epoch and writes:

`Wind_Wave/outputs/logs/training_curve.png`

## Scope

This change adds experiment diagnostics only. It does not change the ConvLSTM
architecture, optimizer, loss, regions, target variables, or default training
duration.

## Success Criteria

1. Dataset samples include a normalized persistence tensor with the same shape as
   targets.
2. Persistence metrics are computed per lead using the same metric functions as
   model predictions.
3. A training run writes validation and test baseline metrics.
4. A training run writes a non-empty training curve PNG.
5. Existing and new tests pass in the `pangu` conda environment.
