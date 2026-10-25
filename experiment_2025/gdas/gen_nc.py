"""Compatibility entry point for batch conversion of downloaded GDAS files."""

import sys
from pathlib import Path

try:
    from .cli import main
except ImportError:  # Direct execution: python experiment_2025/gdas/gen_nc.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiment_2025.gdas.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["process", *sys.argv[1:]]))

