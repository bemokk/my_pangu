# Buoy Plot Chinese Typography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make six Buoy plotting scripts use locally configurable 25%-larger Chinese typography while removing whole-figure titles and footer notes.

**Architecture:** Each target plotting script owns its `FONT_SCALE`, `FONT_FAMILY`, base font sizes, and `TEXT_LABELS` dictionary. Plotting calls read display text from those local dictionaries; data loading, statistics, file names, and output paths remain unchanged.

**Tech Stack:** Python, matplotlib, pytest, AST-based source checks

---

### Task 1: Add Typography Contract Regression Test

**Files:**
- Create: `tests/test_buoy_plot_chinese_typography.py`

- [ ] Add an AST-based test that requires all six target scripts to define `FONT_SCALE = 1.25`, `FONT_FAMILY`, and `TEXT_LABELS`.
- [ ] Add checks that whole-figure `suptitle` and `fig.text` calls are absent.
- [ ] Add checks that scripts with experiment legends expose editable Chinese experiment names.
- [ ] Run `conda run -n pangu python -m pytest tests/test_buoy_plot_chinese_typography.py -q` and verify it fails because the configuration is not implemented yet.

### Task 2: Update General Wind-Speed Plots

**Files:**
- Modify: `Buoy/plots/plot_china_sea_hex_counts.py`
- Modify: `Buoy/plots/plot_wind_speed_metrics_figure2.py`

- [ ] Add local typography configuration and apply scaled font sizes.
- [ ] Replace configurable display labels with Chinese labels.
- [ ] Remove the whole-figure title and footer note.
- [ ] Keep statistical logic and output paths unchanged.

### Task 3: Update Beaufort Plots

**Files:**
- Modify: `Buoy/plots/plot_wind_speed_beaufort_metrics.py`
- Modify: `Buoy/plots/plot_wind_speed_beaufort_sample_counts.py`

- [ ] Add local typography configuration and apply scaled font sizes.
- [ ] Translate experiment names, axes, and panel lead-time titles.
- [ ] Remove whole-figure titles and footer notes.
- [ ] Keep Beaufort grouping, statistics, layouts, and output paths unchanged.

### Task 4: Update Spatial and Track Plots

**Files:**
- Modify: `Buoy/plots/plot_spatial_hex_best_rmse.py`
- Modify: `Buoy/plots/plot_wipha_track_forecast_error.py`

- [ ] Add local typography configuration and apply scaled font sizes.
- [ ] Translate experiment names, state labels, axes, and panel titles.
- [ ] Preserve the existing spatial layout and legend placement.
- [ ] Remove the track plot whole-figure title.
- [ ] Keep statistical and track-error logic unchanged.

### Task 5: Verify Requirements and Regressions

**Files:**
- Verify: all six target scripts and associated tests

- [ ] Run the new typography contract test and verify it passes.
- [ ] Run existing tests for wind-speed metrics, Beaufort plots, spatial hex RMSE, and Wipha case analysis.
- [ ] Run `python -m py_compile` on all six target scripts.
- [ ] Run `git diff --check` and inspect the diff for unintended data/statistics/output-path changes.
- [ ] Commit the implementation after all fresh verification commands pass.
