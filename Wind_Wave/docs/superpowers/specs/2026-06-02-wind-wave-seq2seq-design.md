# Wind-Wave Seq2Seq Design

## Goal

Build a sequence-to-sequence experiment under `E:\PyCharm_WorkSpace\pangu\Wind_Wave` that uses recent ERA5 wind history to predict future multi-lead ERA5 wave fields.

The selected first implementation is a ConvLSTM encoder with separate CNN prediction heads for each forecast lead. This replaces the earlier same-time diagnostic baseline with a real forecast framing while keeping the model small enough to run as an experiment.

## Selected Experiment

Each sample is centered on an initialization time `t0`.

Input wind sequence:

- variables: `u10`, `v10`
- window: `t0 - 23h` through `t0`
- length: 24 hourly frames
- tensor shape before batching: `[24, 2, H, W]`

Wave targets:

- lead times: `+6h`, `+12h`, `+24h`, `+48h`, `+72h`
- variables per lead: `swh`, `mwp`, `peak_wave_period`, `sin_mwd`, `cos_mwd`
- tensor shape before batching: `[5, 5, H, W]`

Wave direction is represented as sine and cosine channels, not raw degrees, so circular direction errors such as 359 degrees versus 1 degree are handled correctly.

If an extracted ERA5 wave file does not contain `peak_wave_period` or `pp1d`, the first experiment keeps the five-channel target shape by using `mwp` as the `peak_wave_period` fallback and emitting a runtime warning. This keeps the training pipeline executable on the downloaded data while making the limitation visible in logs.

## Existing Data

Raw data is already present:

```text
Wind_Wave/
  data/
    2025/
      *.zip
```

Each zip archive contains:

- `data_stream-oper_stepType-instant.nc`
- `data_stream-wave_stepType-instant.nc`

The expected split is:

- `oper`: ERA5 10m wind fields
- `wave`: ERA5 wave fields

No raw data should be moved or deleted. Extracted files, indexes, model outputs, logs, and sample artifacts must stay under `Wind_Wave`.

## Alternatives Considered

1. ConvLSTM encoder with multi-lead CNN heads. This is selected because it models temporal wind evolution directly and keeps the first experiment manageable.
2. Channel-stacked CNN or U-Net. This is simpler and faster, but it treats the 24-hour wind history as unordered channels and has weaker temporal inductive bias.
3. Transformer or PredRNN-style model. This has more modeling capacity, but it is too heavy for the first local experiment and needs more tuning.

## Data Flow

### Archive Handling

The pipeline scans `Wind_Wave/data/2025/*.zip` and extracts archives into:

```text
Wind_Wave/
  data/
    extracted/
      2025/
        archive_stem_folders/
          data_stream-oper_stepType-instant.nc
          data_stream-wave_stepType-instant.nc
```

Each archive is extracted into a folder named after the zip file stem. Extraction is idempotent. If both NetCDF files already exist for a zip stem, extraction is skipped.

### Dataset Index

The index builder opens all extracted quarterly pairs with xarray and builds one chronological time index across 2025. A valid training sample requires:

- wind data for all 24 input hours from `t0 - 23h` to `t0`
- wave data for all configured target leads from `t0 + 6h` through `t0 + 72h`
- matching latitude and longitude grids between wind and wave files

Samples that do not satisfy the full input and target time coverage are skipped. This removes early-year samples without enough wind history and late-year samples without enough future target coverage.

### Variable Mapping

The loader accepts short names and common ERA5 long names.

Wind input variables:

- `u10`, `10m_u_component_of_wind`
- `v10`, `10m_v_component_of_wind`

Wave target variables:

- `swh`, `significant_height_of_combined_wind_waves_and_swell`
- `mwp`, `mean_wave_period`
- `pp1d`, `peak_wave_period`, `peak_wave_period_of_combined_wind_waves_and_swell`, with `mwp` fallback when absent
- `mwd`, `mean_wave_direction`

The model target uses `sin_mwd` and `cos_mwd`, computed from `mwd` degrees with meteorological convention preserved as a circular scalar target.

### Splits

Use chronological splits to avoid future leakage:

- train: first 70 percent of valid initialization times
- validation: next 15 percent
- test: final 15 percent

Normalization statistics are computed from training samples only.

### Spatial Size Controls

Full-resolution ERA5 fields are large, so the first experiment supports controls that make smoke runs practical:

- `--max-samples`: cap the number of valid samples loaded into an experiment
- `--spatial-stride`: use every Nth latitude and longitude point
- `--crop-size`: optional fixed-size spatial crop for quick training

The default smoke command should use a small sample cap and downsampling. A larger training run can reduce the stride after the pipeline is verified.

## Model Architecture

### Encoder

The encoder is a stack of ConvLSTM cells. At each input hour, it receives a normalized wind tensor `[B, 2, H, W]` and updates hidden spatial states. After 24 hours, the final hidden state summarizes the wind evolution leading into `t0`.

### Multi-Lead Heads

The final hidden state is passed to one CNN head per target lead:

- head 1 predicts `+6h`
- head 2 predicts `+12h`
- head 3 predicts `+24h`
- head 4 predicts `+48h`
- head 5 predicts `+72h`

Each head outputs 5 channels:

- `swh`
- `mwp`
- `peak_wave_period`
- `sin_mwd`
- `cos_mwd`

The heads are independent in the first version. This keeps implementation clear and allows per-lead metrics.

### Loss

Use masked mean squared error over finite target values.

The first version uses the same loss weight for all target channels. Direction channels are trained with MSE on `sin_mwd` and `cos_mwd`.

## Training CLI

From the repository root:

```powershell
python Wind_Wave/train.py --epochs 1 --batch-size 1 --max-samples 16 --spatial-stride 8
```

From inside `Wind_Wave`:

```powershell
python -m wind_wave.train --epochs 1 --batch-size 1 --max-samples 16 --spatial-stride 8
```

Default configuration:

- input history: 24 hours
- lead times: `6,12,24,48,72`
- input channels: 2
- target channels per lead: 5
- device: CUDA when available, otherwise CPU

Outputs:

```text
Wind_Wave/
  outputs/
    checkpoints/
      seq2seq_convlstm_best.pt
      seq2seq_convlstm_latest.pt
    logs/
      train_log.csv
      metrics_by_lead.csv
    samples/
      predictions_preview.npz
      sample_metadata.csv
    normalization.json
```

## Evaluation

Report metrics overall and per lead:

- RMSE for `swh`
- RMSE for `mwp`
- RMSE for `peak_wave_period`
- circular MAE for `mwd`, reconstructed from predicted and target sine/cosine channels

Validation metrics are written after each epoch. Test metrics are produced by a separate evaluation command after training.

## Error Handling

The pipeline should fail with clear messages when:

- `Wind_Wave/data/2025` does not exist
- no zip archives are present
- a zip archive does not contain both expected NetCDF files
- required wind variables cannot be found
- required wave variables `swh`, `mwp`, or `mwd` cannot be found
- wind and wave latitude/longitude grids cannot be aligned
- no valid `t0` samples remain after applying history and lead-time requirements
- normalization statistics cannot be computed from finite values
- model output shape does not match `[B, 5, 5, H, W]`

Optional variables not used in this first experiment should not block training.

## Testing

Tests should avoid requiring the full 2025 dataset. Synthetic xarray datasets can verify behavior.

Planned tests:

- variable candidate matching accepts short and long ERA5 names
- direction conversion maps degrees to stable sine/cosine channels
- valid sample indexing requires complete 24-hour history and all configured future leads
- chronological split preserves time order and has no overlap
- spatial stride returns expected reduced grid shape
- synthetic dataset item returns input shape `[24, 2, H, W]` and target shape `[5, 5, H, W]`
- ConvLSTM model maps `[B, 24, 2, H, W]` to `[B, 5, 5, H, W]`
- masked MSE ignores NaN targets and rejects all-NaN targets

Implementation should use test-first development for the data indexer, direction transform, model shape, and loss behavior.

## Out of Scope

The first seq2seq experiment will not:

- use wave history as model input
- predict every hourly lead up to 72 hours
- use a Transformer model
- implement distributed training
- download additional data
- tune hyperparameters extensively
- delete or move existing raw zip archives
- write code, data, or outputs outside `Wind_Wave`

## Success Criteria

The experiment is successful when:

1. A smoke training run completes with `--max-samples 16 --spatial-stride 8`.
2. The dataset creates valid samples using 24 hours of wind history and five future wave leads.
3. The ConvLSTM model produces output shape `[B, 5, 5, H, W]`.
4. Training writes checkpoints, logs, normalization stats, and metrics under `Wind_Wave/outputs`.
5. Unit tests for indexing, transforms, model shape, and loss pass.
