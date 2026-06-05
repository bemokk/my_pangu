# Wind-Wave Regional Seq2Seq Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the Wind_Wave seq2seq pipeline to use Northwest Pacific wind input and China-near-sea wave output with four target channels.

**Architecture:** Add a region parser/selector, pass separate input/output regions into normalization and dataset sampling, and let the ConvLSTM model resize encoded wind features to the target grid before multi-lead heads. Remove `pp1d` from targets, metrics, docs, and tests.

**Tech Stack:** Python 3.10 in conda env `pangu`, PyTorch, xarray, numpy, pandas, pytest.

---

## Tasks

- [ ] Add failing tests for region parsing, regional dataset shapes, four target channels, and model output resizing.
- [ ] Implement region parsing and xarray region selection using `south,north,west,east`.
- [ ] Change dataset targets to `swh`, `mwp`, `cos_mwd`, `sin_mwd` and remove `pp1d` fallback.
- [ ] Update normalization stats for 2 input channels and 4 target channels.
- [ ] Update ConvLSTM model forward to accept `output_size` and return `[B, lead_count, 4, H_out, W_out]`.
- [ ] Update training/evaluation defaults, checkpoint metadata, and metrics CSV fields.
- [ ] Run `conda run -n pangu python -m pytest Wind_Wave/tests -q`.
- [ ] Run smoke training with `conda run -n pangu python Wind_Wave/train.py --epochs 1 --batch-size 1 --max-samples 16 --spatial-stride 8 --num-workers 0`.
- [ ] Commit only `Wind_Wave` docs, tests, and source changes.
