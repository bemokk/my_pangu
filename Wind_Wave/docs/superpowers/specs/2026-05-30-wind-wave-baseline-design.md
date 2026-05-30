# Wind-Wave Baseline Design

## Goal

Build the fastest experimental baseline for training a model that maps ERA5 wind fields to ERA5 wave fields, with all project code, docs, outputs, and data stored under `E:\PyCharm_WorkSpace\pangu\Wind_Wave`.

This first baseline is a same-time diagnostic model: wind at time `t` predicts wave variables at the same time `t`. It is not yet a physically complete forecast system, but it proves the full local workflow from zipped ERA5 data to a saved PyTorch model.

## Existing Context

`Wind_Wave` currently contains `data/2025` with four large zip archives. Each archive contains two NetCDF files:

- `data_stream-oper_stepType-instant.nc`
- `data_stream-wave_stepType-instant.nc`

The expected split is that `oper` contains 10m wind variables and `wave` contains wave variables. The repository root already has ERA5/Pangu utilities, but there is no existing wind-to-wave training pipeline under `Wind_Wave`. The root `.gitignore` ignores `Wind_Wave/`, so future commits for this subproject need explicit force-adds or a later ignore-rule adjustment.

## Recommended Approach

Use a minimal PyTorch CNN baseline.

Inputs:

- `u10`: 10m u-component of wind
- `v10`: 10m v-component of wind

Targets for the first training loop:

- `swh`: significant height of combined wind waves and swell
- `mwp`: mean wave period

Wave direction is intentionally deferred from the first baseline because direction is circular. A later extension should represent wave direction as `sin(direction)` and `cos(direction)` rather than training directly on degrees.

## Alternatives Considered

1. Same-time CNN baseline. Fastest to implement and validate. It proves data loading, normalization, model execution, checkpointing, and logging.
2. Same-time CNN with derived wind speed/direction features and circular wave-direction targets. More useful scientifically, but slightly more preprocessing and loss handling.
3. Sequence-to-sequence forecast model using past wind fields to predict future wave fields. Best match for real wave forecasting, but not appropriate for the fastest experiment.

The selected approach is option 1.

## Data Layout

The project will keep raw downloaded archives in place:

```text
Wind_Wave/
  data/
    2025/
      *.zip
```

Derived local files will stay under `Wind_Wave`:

```text
Wind_Wave/
  data/
    extracted/
      2025/
        <zip-stem>/
          data_stream-oper_stepType-instant.nc
          data_stream-wave_stepType-instant.nc
  outputs/
    checkpoints/
    logs/
    samples/
```

The raw zip files are not deleted. Extraction is idempotent: if both NetCDF files already exist for an archive, extraction is skipped.

## Components

### Configuration

A small configuration module defines paths, variable candidates, train defaults, and output locations. Paths are resolved relative to `Wind_Wave` so the scripts can be run from the project root or from inside the subproject.

### Archive Extraction

The extraction module scans `data/2025/*.zip`, creates one extraction folder per zip stem, and extracts only when needed. It validates that each extracted folder contains both expected NetCDF files.

### ERA5 Dataset Discovery

The data discovery layer builds a list of quarterly pairs:

- one `oper` NetCDF path
- one `wave` NetCDF path

It opens these with xarray and identifies coordinates using common ERA5 coordinate names:

- time: `valid_time` or `time`
- latitude: `latitude` or `lat`
- longitude: `longitude` or `lon`

It identifies variables by short names or long ERA5 names:

- wind: `u10`, `10m_u_component_of_wind`; `v10`, `10m_v_component_of_wind`
- wave: `swh`, `significant_height_of_combined_wind_waves_and_swell`; `mwp`, `mean_wave_period`

### Normalization

Normalization statistics are computed from the training split only. For the fastest experiment, compute per-variable mean and standard deviation over a capped number of samples if `--max-samples` is provided. Save the result as JSON under `outputs/normalization.json`.

### PyTorch Dataset

The dataset exposes one sample per aligned time index. Each item returns:

- input tensor: shape `[2, H, W]`
- target tensor: shape `[2, H, W]`
- timestamp metadata for logging or sample export

The first version may load time slices lazily from xarray. It should avoid reading all 2025 data into memory at once.

### Model

Use a compact convolutional encoder-decoder:

- input channels: 2
- output channels: 2
- several Conv2d/ReLU blocks
- downsample and upsample with simple interpolation or stride/pooling

This is intentionally modest. The goal is to run a short baseline quickly, not to optimize final forecast skill.

### Training Script

The CLI entry point trains the model and writes artifacts. From the repository root:

```powershell
python Wind_Wave/train.py --epochs 1 --batch-size 1 --max-samples 32
```

From inside `Wind_Wave`:

```powershell
python -m wind_wave.train --epochs 1 --batch-size 1 --max-samples 32
```

Outputs:

- best checkpoint: `outputs/checkpoints/baseline_cnn.pt`
- latest checkpoint: `outputs/checkpoints/latest.pt`
- training log CSV: `outputs/logs/train_log.csv`
- normalization stats: `outputs/normalization.json`

The script selects CUDA if available, otherwise CPU.

### Smoke Run

A short smoke run with `--max-samples 8 --epochs 1` must complete without reading the full dataset into memory. This verifies data extraction, xarray loading, tensor conversion, forward pass, loss calculation, backward pass, and checkpoint writing.

## Error Handling

The pipeline should fail with clear messages when:

- `data/2025` does not exist
- no zip archives are present
- an archive does not contain the two expected NetCDF files
- required wind or wave variables cannot be found
- time coordinates between `oper` and `wave` cannot be aligned
- tensors contain no finite values after preprocessing

For missing or unsupported optional variables, the first baseline should not fail. It only requires `u10`, `v10`, `swh`, and `mwp`.

## Testing

Tests focus on the local code behavior and avoid requiring the full 40GB dataset.

Planned tests:

- variable candidate matching works for short and long names
- coordinate normalization accepts `time` and `valid_time`
- archive extraction skips already extracted files
- synthetic xarray datasets produce tensors with expected shapes
- normalization handles finite values and rejects all-NaN arrays
- model forward pass maps `[B, 2, H, W]` to `[B, 2, H, W]`

Implementation should follow test-first development where practical.

## Out of Scope

The first baseline will not:

- train a true future forecast model
- predict wave direction directly
- optimize hyperparameters
- implement distributed training
- download additional data
- delete or move existing raw zip archives

## Success Criteria

The experiment is successful when:

1. Running the training command from `E:\PyCharm_WorkSpace\pangu` reads local `Wind_Wave/data/2025` archives.
2. A smoke training run completes on a small sample cap.
3. Model checkpoints and logs are written under `Wind_Wave/outputs`.
4. The code does not store generated artifacts outside `Wind_Wave`.
5. Tests for data utilities and model shape pass.
