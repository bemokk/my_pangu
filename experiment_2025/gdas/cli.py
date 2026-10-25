"""Command line entry point for the experiment_2025 GDAS pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import download_and_prepare, download_range, process_existing, plan


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use YYYY-MM-DDTHH:MM[:SS] UTC") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare GDAS/FNL Pangu inputs")
    parser.add_argument("action", choices=("run", "download", "process"), nargs="?", default="run")
    parser.add_argument("--start", "--start-date", required=True, type=_parse_time)
    parser.add_argument("--end", "--end-date", required=True, type=_parse_time)
    parser.add_argument("--interval-hours", type=int, default=24, help="Default: one cycle per day at the start hour")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show date coverage and year-based storage paths without downloading or writing")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    kwargs = dict(interval_hours=args.interval_hours, raw_root=args.raw_root, output_root=args.output_root,
                  overwrite=args.overwrite, continue_on_error=args.continue_on_error)
    try:
        if args.dry_run:
            plan(args.start, args.end, **kwargs)
            return 0
        if args.action == "process":
            outputs = process_existing(args.start, args.end, **kwargs)
        else:
            if args.base_url:
                kwargs["base_url"] = args.base_url
            if args.action == "download":
                kwargs.pop("overwrite")
                kwargs.pop("output_root")
                outputs = download_range(args.start, args.end, **kwargs)
            else:
                outputs = download_and_prepare(args.start, args.end, **kwargs)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[GDAS] {exc}")
        return 1
    print(f"[GDAS] {args.action} complete: {len(outputs)} cycle(s)")
    for output in outputs:
        print(f"  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

