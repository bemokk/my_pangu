# Wind-Wave Baseline and Training Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistence baseline and a loss-curve artifact to the regional wind-wave seq2seq experiment.

**Architecture:** The dataset produces a normalized persistence tensor from the wave field at initialization time. Shared evaluation helpers calculate model and persistence metrics consistently, while a focused plotting helper renders the existing epoch log.

**Tech Stack:** Python 3.10, PyTorch, xarray, NumPy, pandas, Pillow, pytest.

---

## Tasks

- [x] Add a failing dataset test requiring persistence forecasts to repeat the `t0` wave field for every lead.
- [x] Implement normalized persistence output in `WindWaveSeq2SeqDataset`.
- [x] Add failing tests for persistence evaluation metrics and training-curve output.
- [x] Refactor shared per-lead metric calculation and implement persistence loader evaluation.
- [x] Implement training-curve PNG generation.
- [x] Write validation and test persistence metrics during training.
- [x] Run the full test suite in conda environment `pangu`.
- [x] Run a four-sample real-data smoke training and inspect generated diagnostics.
- [x] Commit and push the design, plan, source, and tests.
