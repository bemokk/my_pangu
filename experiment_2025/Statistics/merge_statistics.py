"""Merge per-experiment CSV outputs into the requested Statistics/data folder."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FILES = (
    "wind_speed_metrics_by_lead_month.csv",
    "wind_direction_metrics_by_lead_month.csv",
    "wind_speed_metrics_by_lead.csv",
    "wind_direction_metrics_by_lead.csv",
    "wind_speed_metrics_by_month.csv",
    "wind_direction_metrics_by_month.csv",
    "missing_outputs.csv",
)


def read_rows(path: Path):
    if not path.is_file():
        return [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames or []


def write_rows(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", type=Path, default=Path(__file__).resolve().parent / "data" / "_per_experiment")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args(argv)
    parts = sorted(path for path in args.parts_root.iterdir() if path.is_dir())
    if not parts:
        raise SystemExit(f"No per-experiment folders found: {args.parts_root}")
    for filename in FILES:
        rows = []
        fields = []
        for part in parts:
            part_rows, part_fields = read_rows(part / filename)
            rows.extend(part_rows)
            for field in part_fields:
                if field not in fields:
                    fields.append(field)
        if rows:
            write_rows(args.output_root / filename, rows, fields)
    summaries = []
    for part in parts:
        path = part / "run_summary.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    summary = {
        "parts": [path.name for path in parts],
        "experiments": [item for item in summaries for item in item.get("experiments", [])],
        "summaries": summaries,
    }
    (args.output_root / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {len(parts)} experiments into {args.output_root}")


if __name__ == "__main__":
    raise SystemExit(main())
