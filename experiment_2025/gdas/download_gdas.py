"""Backward-compatible command line name for :mod:`experiment_2025.gdas`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .cli import main
except ImportError:  # Direct execution: python experiment_2025/gdas/download_gdas.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiment_2025.gdas.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))

