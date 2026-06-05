# Wind-Wave Regional Seq2Seq Design

## Goal

Change the seq2seq experiment to predict China-near-sea wave fields from a larger Northwest Pacific wind context, using only the variables available in the downloaded ERA5 wave archives.

## Selected Experiment

Input wind sequence:

- history: 24 hourly frames from `t0 - 23h` through `t0`
- variables: `u10`, `v10`
- region: `5-45N, 95-150E`

Output wave sequence:

- leads: `+6h`, `+12h`, `+24h`, `+48h`, `+72h`
- variables: `swh`, `mwp`, `cos_mwd`, `sin_mwd`
- region: `15-40N, 105-135E`

Peak wave period is removed. There is no `pp1d` fallback and no peak-period metric.

## Architecture Changes

The dataset selects different spatial regions for inputs and targets. This means input tensors and target tensors can have different height and width.

The ConvLSTM encoder consumes the larger wind-context grid. Before each lead head predicts wave variables, the encoded hidden state is resized to the target wave grid size with bilinear interpolation. Each head then predicts 4 channels on the China-near-sea grid.

## CLI Defaults

The training and evaluation CLIs keep the existing lead-time defaults and add regional defaults:

- `--input-region 5,45,95,150`
- `--output-region 15,40,105,135`

The region format is `south,north,west,east` in degrees.

## Metrics

Metrics are reported per lead:

- `rmse_swh`
- `rmse_mwp`
- `mae_mwd_degrees`, reconstructed from `cos_mwd` and `sin_mwd`

## Success Criteria

1. Dataset samples return input shape `[24, 2, H_in, W_in]` and target shape `[5, 4, H_out, W_out]`.
2. The model accepts input and an output size, returning `[B, 5, 4, H_out, W_out]`.
3. Training and evaluation run with the regional defaults.
4. Tests pass in the `pangu` conda environment.
5. A smoke training run writes updated checkpoints, logs, normalization stats, and preview samples under `Wind_Wave/outputs`.
