# Wind-Wave Seq2Seq Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a ConvLSTM sequence-to-sequence wind-to-wave experiment that trains from local ERA5 zip archives in `Wind_Wave/data/2025`.

**Architecture:** The package `Wind_Wave/wind_wave` owns focused modules for paths, archive extraction, ERA5 variable matching, time indexing, PyTorch datasets, model, loss, metrics, training, and evaluation. `Wind_Wave/train.py` and `Wind_Wave/evaluate.py` are thin root-level wrappers so commands work from the repository root while outputs remain under `Wind_Wave`.

**Tech Stack:** Python 3.10 in conda env `pangu`, PyTorch, xarray, numpy, pandas, pytest.

---

## File Structure

- Create: `Wind_Wave/wind_wave/__init__.py` package marker.
- Create: `Wind_Wave/wind_wave/config.py` project paths, constants, and default experiment settings.
- Create: `Wind_Wave/wind_wave/extract.py` zip extraction and extracted pair discovery.
- Create: `Wind_Wave/wind_wave/era5.py` xarray coordinate normalization, variable matching, direction transforms, and synthetic dataset helpers for tests.
- Create: `Wind_Wave/wind_wave/indexing.py` valid `t0` sample construction and chronological splitting.
- Create: `Wind_Wave/wind_wave/dataset.py` PyTorch dataset, normalization stats, spatial stride, and crop support.
- Create: `Wind_Wave/wind_wave/model.py` ConvLSTM cell, encoder, and multi-lead heads.
- Create: `Wind_Wave/wind_wave/losses.py` masked MSE.
- Create: `Wind_Wave/wind_wave/metrics.py` RMSE and circular wave-direction MAE.
- Create: `Wind_Wave/wind_wave/train.py` training CLI and checkpoint/log writing.
- Create: `Wind_Wave/wind_wave/evaluate.py` evaluation CLI for saved checkpoints.
- Create: `Wind_Wave/train.py` repository-root wrapper for training.
- Create: `Wind_Wave/evaluate.py` repository-root wrapper for evaluation.
- Create: `Wind_Wave/tests/test_era5.py` unit tests for variables, coordinates, and direction transforms.
- Create: `Wind_Wave/tests/test_indexing.py` unit tests for valid sample indexing and splits.
- Create: `Wind_Wave/tests/test_dataset.py` unit tests for tensor shapes, stride, and normalization.
- Create: `Wind_Wave/tests/test_model.py` unit tests for model output shape.
- Create: `Wind_Wave/tests/test_losses_metrics.py` unit tests for masked loss and metrics.

### Task 1: ERA5 Utilities And Sample Index

**Files:**
- Create: `Wind_Wave/wind_wave/__init__.py`
- Create: `Wind_Wave/wind_wave/config.py`
- Create: `Wind_Wave/wind_wave/era5.py`
- Create: `Wind_Wave/wind_wave/indexing.py`
- Test: `Wind_Wave/tests/test_era5.py`
- Test: `Wind_Wave/tests/test_indexing.py`

- [ ] **Step 1: Write failing tests for ERA5 utilities**

Create `Wind_Wave/tests/test_era5.py` with:

```python
import numpy as np
import pandas as pd
import xarray as xr

from wind_wave.era5 import (
    direction_degrees_to_unit,
    find_data_var,
    normalize_time_coord,
)


def test_find_data_var_accepts_short_and_long_names():
    ds = xr.Dataset(
        {
            "10m_u_component_of_wind": (("time", "latitude", "longitude"), np.zeros((1, 2, 2))),
            "swh": (("time", "latitude", "longitude"), np.ones((1, 2, 2))),
        }
    )

    assert find_data_var(ds, ["u10", "10m_u_component_of_wind"]) == "10m_u_component_of_wind"
    assert find_data_var(ds, ["swh", "significant_height_of_combined_wind_waves_and_swell"]) == "swh"


def test_normalize_time_coord_renames_valid_time():
    ds = xr.Dataset(
        {"u10": (("valid_time",), np.array([1.0]))},
        coords={"valid_time": pd.date_range("2025-01-01", periods=1, freq="h")},
    )

    normalized = normalize_time_coord(ds)

    assert "time" in normalized.dims
    assert "valid_time" not in normalized.dims


def test_direction_degrees_to_unit_wraps_around():
    sin_v, cos_v = direction_degrees_to_unit(np.array([0.0, 90.0, 360.0]))

    np.testing.assert_allclose(sin_v, np.array([0.0, 1.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(cos_v, np.array([1.0, 0.0, 1.0]), atol=1e-6)
```

- [ ] **Step 2: Write failing tests for sample indexing**

Create `Wind_Wave/tests/test_indexing.py` with:

```python
import pandas as pd

from wind_wave.indexing import build_valid_initialization_times, chronological_split


def test_valid_initialization_times_require_history_and_future_leads():
    times = pd.date_range("2025-01-01T00:00", periods=120, freq="h")

    valid = build_valid_initialization_times(
        wind_times=times,
        wave_times=times,
        history_hours=24,
        lead_hours=(6, 12, 24, 48, 72),
    )

    assert valid[0] == pd.Timestamp("2025-01-01T23:00")
    assert valid[-1] == pd.Timestamp("2025-01-02T23:00")
    assert len(valid) == 25


def test_chronological_split_preserves_order_and_no_overlap():
    times = pd.date_range("2025-01-01", periods=20, freq="h")

    train, val, test = chronological_split(times, train_fraction=0.7, val_fraction=0.15)

    assert len(train) == 14
    assert len(val) == 3
    assert len(test) == 3
    assert train[-1] < val[0] < test[0]
```

- [ ] **Step 3: Run tests and verify red**

Run:

```powershell
conda run -n pangu python -m pytest Wind_Wave/tests/test_era5.py Wind_Wave/tests/test_indexing.py -q
```

Expected: FAIL because the `wind_wave` modules do not exist yet.

- [ ] **Step 4: Implement minimal ERA5 utilities and indexing**

Create `config.py`, `era5.py`, and `indexing.py`.

Required public functions and behavior:

- `project_root`: returns the absolute `Wind_Wave` directory.
- `find_data_var`: returns the first candidate present in `ds.data_vars`, then checks xarray attributes `shortName`, `GRIB_shortName`, `standard_name`, and `long_name`; raises `KeyError` when no candidate is present.
- `normalize_time_coord`: renames `valid_time` to `time`, expands scalar time coordinates into a one-item dimension, and sorts by time.
- `align_wind_to_wave_grid`: selects the finer wind grid onto the wave dataset latitude and longitude coordinates.
- `direction_degrees_to_unit`: converts degrees to radians with `np.deg2rad`, then returns float32 sine and cosine arrays.
- `build_valid_initialization_times`: returns every `t0` with complete hourly wind history from `t0 - 23h` through `t0` and every configured wave lead.
- `chronological_split`: returns three ordered lists with no overlap and raises `ValueError` when any split would be empty.

- [ ] **Step 5: Run tests and verify green**

Run:

```powershell
conda run -n pangu python -m pytest Wind_Wave/tests/test_era5.py Wind_Wave/tests/test_indexing.py -q
```

Expected: PASS.

### Task 2: Archive Extraction, Dataset, And Normalization

**Files:**
- Create: `Wind_Wave/wind_wave/extract.py`
- Create: `Wind_Wave/wind_wave/dataset.py`
- Test: `Wind_Wave/tests/test_dataset.py`

- [ ] **Step 1: Write failing dataset tests**

Create `Wind_Wave/tests/test_dataset.py` with:

```python
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wind_wave.dataset import (
    NormalizationStats,
    WindWaveSeq2SeqDataset,
    compute_normalization_stats,
)


def make_synthetic_pair():
    times = pd.date_range("2025-01-01", periods=120, freq="h")
    lat = np.linspace(10.0, 8.0, 6)
    lon = np.linspace(120.0, 123.0, 8)
    shape = (len(times), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    wind = xr.Dataset(
        {
            "u10": (("time", "latitude", "longitude"), base),
            "v10": (("time", "latitude", "longitude"), base + 1.0),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    wave = xr.Dataset(
        {
            "swh": (("time", "latitude", "longitude"), base + 2.0),
            "mwp": (("time", "latitude", "longitude"), base + 3.0),
            "pp1d": (("time", "latitude", "longitude"), base + 4.0),
            "mwd": (("time", "latitude", "longitude"), np.full(shape, 90.0, dtype=np.float32)),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    return wind, wave


def test_dataset_returns_expected_seq2seq_shapes_with_stride():
    wind, wave = make_synthetic_pair()
    stats = NormalizationStats.identity(input_names=("u10", "v10"), target_names=("swh", "mwp", "pp1d", "sin_mwd", "cos_mwd"))
    ds = WindWaveSeq2SeqDataset(
        wind,
        wave,
        initialization_times=[pd.Timestamp("2025-01-02T00:00")],
        stats=stats,
        spatial_stride=2,
    )

    sample = ds[0]

    assert sample["inputs"].shape == (24, 2, 3, 4)
    assert sample["targets"].shape == (5, 5, 3, 4)
    assert sample["target_times"][0] == "2025-01-02T06:00:00"


def test_compute_normalization_stats_rejects_all_nan_values():
    wind, wave = make_synthetic_pair()
    wind["u10"][:] = np.nan

    with pytest.raises(ValueError, match="finite"):
        compute_normalization_stats(
            wind,
            wave,
            initialization_times=[pd.Timestamp("2025-01-02T00:00")],
            spatial_stride=2,
        )
```

- [ ] **Step 2: Run dataset tests and verify red**

Run:

```powershell
conda run -n pangu python -m pytest Wind_Wave/tests/test_dataset.py -q
```

Expected: FAIL because `wind_wave.dataset` does not exist.

- [ ] **Step 3: Implement extraction and dataset**

Implement these public objects:

- `ExtractedPair`: frozen dataclass with `archive`, `extract_dir`, `oper_nc`, and `wave_nc`.
- `discover_archives`: returns sorted `.zip` paths and raises `FileNotFoundError` when the raw directory or archives are absent.
- `extract_archives`: extracts into archive-stem folders and validates the expected `oper` and `wave` NetCDF files.
- `NormalizationStats`: frozen dataclass with `input_mean`, `input_std`, `target_mean`, `target_std`, `input_names`, `target_names`, plus an `identity` classmethod for tests.
- `compute_normalization_stats`: computes finite means and standard deviations from training initialization times, spatial stride, and optional crop.
- `WindWaveSeq2SeqDataset`: returns a dict with `inputs`, `targets`, `t0`, `input_times`, and `target_times`. Inputs are `[24, 2, H, W]`; targets are `[5, 5, H, W]`.
- If `pp1d` or `peak_wave_period` is absent in the wave dataset, the dataset uses `mwp` for the peak-period target channel and emits a warning.

- [ ] **Step 4: Run dataset tests and verify green**

Run:

```powershell
conda run -n pangu python -m pytest Wind_Wave/tests/test_dataset.py -q
```

Expected: PASS.

### Task 3: ConvLSTM Model, Loss, And Metrics

**Files:**
- Create: `Wind_Wave/wind_wave/model.py`
- Create: `Wind_Wave/wind_wave/losses.py`
- Create: `Wind_Wave/wind_wave/metrics.py`
- Test: `Wind_Wave/tests/test_model.py`
- Test: `Wind_Wave/tests/test_losses_metrics.py`

- [ ] **Step 1: Write failing model and loss tests**

Create `Wind_Wave/tests/test_model.py` with:

```python
import torch

from wind_wave.model import ConvLSTMWindWaveModel


def test_convlstm_model_outputs_multi_lead_wave_tensor():
    model = ConvLSTMWindWaveModel(input_channels=2, hidden_channels=4, lead_count=5, target_channels=5)
    x = torch.randn(2, 24, 2, 8, 10)

    y = model(x)

    assert y.shape == (2, 5, 5, 8, 10)
```

Create `Wind_Wave/tests/test_losses_metrics.py` with:

```python
import math

import torch

from wind_wave.losses import masked_mse_loss
from wind_wave.metrics import circular_mae_degrees


def test_masked_mse_ignores_nan_targets():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, float("nan"), 5.0])

    loss = masked_mse_loss(pred, target)

    assert torch.isclose(loss, torch.tensor(2.0))


def test_masked_mse_rejects_all_nan_targets():
    pred = torch.tensor([1.0])
    target = torch.tensor([float("nan")])

    try:
        masked_mse_loss(pred, target)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_circular_mae_uses_shortest_angle():
    pred_sin = torch.tensor([math.sin(math.radians(359.0))])
    pred_cos = torch.tensor([math.cos(math.radians(359.0))])
    target_sin = torch.tensor([math.sin(math.radians(1.0))])
    target_cos = torch.tensor([math.cos(math.radians(1.0))])

    assert circular_mae_degrees(pred_sin, pred_cos, target_sin, target_cos).item() < 3.0
```

- [ ] **Step 2: Run tests and verify red**

Run:

```powershell
conda run -n pangu python -m pytest Wind_Wave/tests/test_model.py Wind_Wave/tests/test_losses_metrics.py -q
```

Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement model, loss, and metrics**

Implement `ConvLSTMCell`, `ConvLSTMEncoder`, `ConvLSTMWindWaveModel`, `masked_mse_loss`, `rmse`, and `circular_mae_degrees`. The model accepts `[B, 24, 2, H, W]` and returns `[B, 5, 5, H, W]`.

- [ ] **Step 4: Run tests and verify green**

Run:

```powershell
conda run -n pangu python -m pytest Wind_Wave/tests/test_model.py Wind_Wave/tests/test_losses_metrics.py -q
```

Expected: PASS.

### Task 4: Training And Evaluation CLI

**Files:**
- Create: `Wind_Wave/wind_wave/train.py`
- Create: `Wind_Wave/wind_wave/evaluate.py`
- Create: `Wind_Wave/train.py`
- Create: `Wind_Wave/evaluate.py`
- Test: `Wind_Wave/tests/test_cli_imports.py`

- [ ] **Step 1: Write failing CLI import tests**

Create `Wind_Wave/tests/test_cli_imports.py` with:

```python
from wind_wave.train import build_arg_parser as build_train_parser
from wind_wave.evaluate import build_arg_parser as build_eval_parser


def test_train_parser_defaults_match_seq2seq_design():
    args = build_train_parser().parse_args([])

    assert args.history_hours == 24
    assert args.lead_hours == "6,12,24,48,72"
    assert args.batch_size == 1


def test_eval_parser_accepts_checkpoint_argument():
    args = build_eval_parser().parse_args(["--checkpoint", "model.pt"])

    assert args.checkpoint == "model.pt"
```

- [ ] **Step 2: Run CLI tests and verify red**

Run:

```powershell
conda run -n pangu python -m pytest Wind_Wave/tests/test_cli_imports.py -q
```

Expected: FAIL because `wind_wave.train` and `wind_wave.evaluate` do not exist.

- [ ] **Step 3: Implement training and evaluation CLIs**

Training CLI public functions:

- `build_arg_parser`: defines defaults for 24-hour history, `6,12,24,48,72` leads, batch size 1, and output paths under `Wind_Wave/outputs`.
- `train`: extracts archives, opens datasets, builds valid `t0` samples, computes training-only normalization, trains for the requested epochs, writes logs and checkpoints, and returns final metrics.
- `main`: parses optional argv, runs `train`, and returns process code 0.

Evaluation CLI public functions:

- `build_arg_parser`: requires `--checkpoint` and accepts the same data and spatial controls as training.
- `evaluate`: loads a checkpoint, builds the test split, writes per-lead metrics, and returns metric values.
- `main`: parses optional argv, runs `evaluate`, and returns process code 0.

Root wrappers call `raise SystemExit(main())`. Training writes `outputs/checkpoints/seq2seq_convlstm_latest.pt`, `outputs/logs/train_log.csv`, `outputs/logs/metrics_by_lead.csv`, and `outputs/normalization.json`.

- [ ] **Step 4: Run CLI tests and verify green**

Run:

```powershell
conda run -n pangu python -m pytest Wind_Wave/tests/test_cli_imports.py -q
```

Expected: PASS.

### Task 5: Full Verification And Smoke Training

**Files:**
- Modify only files created in Tasks 1 through 4.

- [ ] **Step 1: Run all unit tests**

Run:

```powershell
conda run -n pangu python -m pytest Wind_Wave/tests -q
```

Expected: PASS.

- [ ] **Step 2: Run smoke training in conda pangu**

Run:

```powershell
conda run -n pangu python Wind_Wave/train.py --epochs 1 --batch-size 1 --max-samples 16 --spatial-stride 8 --num-workers 0
```

Expected: command exits 0 and writes files under `Wind_Wave/outputs`.

- [ ] **Step 3: Inspect generated outputs**

Run:

```powershell
Get-ChildItem -Recurse 'Wind_Wave\outputs' | Select-Object FullName,Length | Format-Table -AutoSize
```

Expected: checkpoint, CSV logs, normalization JSON, and sample files exist.

- [ ] **Step 4: Commit Wind_Wave changes only**

Run:

```powershell
git status --short
git add -f Wind_Wave/wind_wave Wind_Wave/tests Wind_Wave/train.py Wind_Wave/evaluate.py Wind_Wave/docs/superpowers/plans/2026-06-02-wind-wave-seq2seq.md
git diff --cached --name-only
git commit -m "feat: add wind wave seq2seq training pipeline"
```

Expected: cached files are only `Wind_Wave` source, tests, wrappers, and plan docs. Existing `Buoy`, `trash`, and typhoon CSV changes remain unstaged.
