# Wind-Wave V2 Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run the v2 minimal closed-loop `M2-wave0-residual` experiment, then run `M2-direct` and `M2-wave0-direct` ablations.

**Architecture:** Extend dataset samples with future target-lead wind and current wave state. Add a v2 model with past-wind ConvLSTM, shared future-wind CNN, optional wave0 CNN, and multi-lead heads that support direct and residual prediction. Keep the original M1 path as the default and write each run under a separate run-name output directory.

**Tech Stack:** Python 3.10, PyTorch, xarray, NumPy, pandas, Pillow, pytest.

---

## Tasks

- [x] Add failing tests for future wind, wave0, v2 model forward paths, CLI variants, and run-specific outputs.
- [x] Implement dataset fields `future_wind` and `wave0`.
- [x] Implement `WindWaveV2Model` with direct and residual prediction modes.
- [x] Add CLI/model factory support for `m2-direct`, `m2-wave0-direct`, and `m2-wave0-residual`.
- [x] Preserve M1 default behavior and existing tests.
- [x] Run full tests in the `pangu` environment.
- [x] Run low-resolution full-data experiments for the three v2 variants.
- [x] Evaluate best checkpoints on the test split.
- [x] Write the Chinese Markdown experiment report to `Wind_Wave/docs`.
- [x] Commit and push Wind_Wave changes.
