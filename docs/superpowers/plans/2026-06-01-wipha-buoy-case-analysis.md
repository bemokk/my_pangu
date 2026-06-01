# Wipha Buoy Case Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Typhoon Wipha 2025 buoy and track verification workflow that produces the requested Figure 10, Figure 11, Table 5, and track-error comparison outputs.

**Architecture:** Add one focused plotting/analysis script under `Buoy/plots/` that reads existing matched buoy samples, fetches or caches NMC Wipha track data, extracts forecast typhoon centers from existing `.npy` surface fields, and writes all case outputs. Add lightweight unit tests for reusable math and aggregation helpers.

**Tech Stack:** Python, pandas, numpy, matplotlib, requests, pytest; direct `.npy` field reading avoids optional netCDF runtime dependencies.

---

## File Structure

- Create `Buoy/plots/plot_wipha_case_analysis.py`: end-to-end Wipha case workflow and reusable helper functions.
- Create `tests/test_wipha_case_analysis.py`: tests for circular direction differences, circular mean aggregation, shortest-lead selection, and haversine distances.
- Write outputs to `Buoy/results/wipha_case/` and `Buoy/results/figures/`.

### Task 1: Add math and aggregation tests

**Files:**
- Create: `tests/test_wipha_case_analysis.py`
- Create later: `Buoy/plots/plot_wipha_case_analysis.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_wipha_case_analysis.py` with imports from `Buoy.plots.plot_wipha_case_analysis` and tests for:

```python
import pandas as pd

from Buoy.plots.plot_wipha_case_analysis import (
    angular_difference_deg,
    circular_mean_deg,
    haversine_km,
    select_shortest_lead_forecasts,
)


def test_angular_difference_wraps_to_shortest_path():
    assert angular_difference_deg(350, 10) == 20
    assert angular_difference_deg(10, 350) == -20
    assert angular_difference_deg(180, 0) == -180


def test_circular_mean_handles_north_wrap():
    assert abs(circular_mean_deg([350, 10]) - 360) < 1e-9 or abs(circular_mean_deg([350, 10])) < 1e-9


def test_haversine_one_degree_equator_is_about_111_km():
    assert 110 <= haversine_km(0, 0, 1, 0) <= 112


def test_select_shortest_lead_forecasts_keeps_one_row_per_dataset_time():
    df = pd.DataFrame(
        {
            "platform_id": ["A", "A", "A"],
            "datetime_utc": pd.to_datetime(["2025-07-17 03:00"] * 3),
            "dataset": ["gdas_forecast", "gdas_forecast", "era5_lagged_5d"],
            "lead_hour": [6, 3, 3],
            "pred_speed_ms": [8.0, 7.0, 6.0],
            "pred_dir_deg": [90.0, 80.0, 70.0],
        }
    )
    out = select_shortest_lead_forecasts(df)
    assert len(out) == 2
    assert out[out["dataset"] == "gdas_forecast"].iloc[0]["lead_hour"] == 3
```

- [ ] **Step 2: Run tests and verify import failure**

Run:

```bash
python -m pytest tests/test_wipha_case_analysis.py -q
```

Expected: fails because `Buoy.plots.plot_wipha_case_analysis` does not exist yet.

### Task 2: Implement helper functions and output workflow

**Files:**
- Create: `Buoy/plots/plot_wipha_case_analysis.py`

- [ ] **Step 1: Implement helpers and constants**

The script must expose:

- `angular_difference_deg(pred, obs)`
- `circular_mean_deg(values)`
- `haversine_km(lon1, lat1, lon2, lat2)`
- `select_shortest_lead_forecasts(df)`

It must define the approved window, selected platforms, dataset labels, output directories, and surface-grid metadata.

- [ ] **Step 2: Implement buoy data preparation**

Read `matched_buoy_model_wind_samples.csv`, filter selected platforms and time window, aggregate observations per platform/time, select shortest lead forecasts per platform/time/dataset, and merge observations with forecasts for plotting and statistics.

- [ ] **Step 3: Implement Wipha real-track loading**

Adapt `trash/get_typhoon_path.py` logic into functions that:

1. Try cached CSVs under `Buoy/results/wipha_case/`.
2. Try NMC metadata search for 2025 Wipha/韦帕.
3. Try direct cached/manual fallback if available.
4. Produce a clear warning and continue with forecast-only tracks if download fails.

- [ ] **Step 4: Implement forecast center extraction from `.npy`**

Read surface arrays from GDAS and ERA5 lagged output paths, crop to a Wipha search box, locate the minimum MSL point, and use previous center to update a small moving search box when possible.

- [ ] **Step 5: Implement plotting and tables**

Write functions for:

- track and buoy-location map,
- wind speed/wind direction time series,
- statistics CSV/XLSX and table figure,
- 72 h track forecast/error comparison figure.

- [ ] **Step 6: Add CLI main**

`python Buoy/plots/plot_wipha_case_analysis.py` should run the full workflow and print all generated paths.

### Task 3: Run validation and generate outputs

**Files:**
- Generated outputs under `Buoy/results/wipha_case/`
- Generated figures under `Buoy/results/figures/`

- [ ] **Step 1: Run unit tests**

```bash
python -m pytest tests/test_wipha_case_analysis.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax check**

```bash
python -m py_compile Buoy/plots/plot_wipha_case_analysis.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Generate Wipha outputs**

```bash
python Buoy/plots/plot_wipha_case_analysis.py
```

Expected: prints generated CSV/PNG/SVG paths and creates the four requested products.

- [ ] **Step 4: Inspect git status**

```bash
git status --short
```

Expected: new script, tests, plan, and generated outputs visible for review.

### Task 4: Commit code changes

**Files:**
- `Buoy/plots/plot_wipha_case_analysis.py`
- `tests/test_wipha_case_analysis.py`
- `docs/superpowers/plans/YYYY-MM-DD-wipha-buoy-case-analysis.md`

- [ ] **Step 1: Commit implementation files**

```bash
git add Buoy/plots/plot_wipha_case_analysis.py tests/test_wipha_case_analysis.py docs/superpowers/plans/YYYY-MM-DD-wipha-buoy-case-analysis.md
git commit -m "feat: add Wipha buoy case analysis workflow"
```

- [ ] **Step 2: Leave generated outputs uncommitted unless requested**

Generated outputs remain available in the working tree for user review.
