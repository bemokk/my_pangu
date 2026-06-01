# Wipha Case Plot Script Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the Wipha case monolithic plotting workflow into independent plot scripts with a shared common module.

**Architecture:** Move shared constants, calculations, data preparation, typhoon path extraction, and statistics into `wipha_case_common.py`. Put each figure/table renderer into its own `plot_wipha_*.py` script. Keep the original script as a thin orchestrator.

**Tech Stack:** Python, pandas, numpy, matplotlib, pytest.

---

### Task 1: Extract common module

**Files:**
- Create: `Buoy/plots/wipha_case_common.py`
- Modify: `tests/test_wipha_case_analysis.py`

- [ ] Move constants and non-plotting functions from `plot_wipha_case_analysis.py` into `wipha_case_common.py`.
- [ ] Update tests to import utilities from `Buoy.plots.wipha_case_common`.

### Task 2: Create focused plotting scripts

**Files:**
- Create: `Buoy/plots/plot_wipha_track_buoy_locations.py`
- Create: `Buoy/plots/plot_wipha_buoy_wind_timeseries.py`
- Create: `Buoy/plots/plot_wipha_buoy_wind_statistics_table.py`
- Create: `Buoy/plots/plot_wipha_track_forecast_error.py`

- [ ] Move each plotting function into the corresponding script.
- [ ] Add a `main()` entry point to each script.
- [ ] Ensure each script only generates its own outputs.

### Task 3: Replace monolithic script with orchestrator

**Files:**
- Modify: `Buoy/plots/plot_wipha_case_analysis.py`

- [ ] Replace implementation with imports from the four focused scripts.
- [ ] `main()` should call each script's `generate()` and print generated paths.

### Task 4: Validate

**Files:**
- Modify: `tests/test_wipha_case_analysis.py`

- [ ] Add importability test for the four plotting scripts.
- [ ] Run `python -m pytest tests/test_wipha_case_analysis.py -q`.
- [ ] Run `python -m py_compile` on all five Wipha scripts plus the common module.
- [ ] Run the orchestrator to confirm all outputs are still generated.
