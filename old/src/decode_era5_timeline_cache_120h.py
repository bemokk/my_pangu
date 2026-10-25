from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "model_output" / "era5"
DEFAULT_START_DATE = datetime(2025, 6, 26)
DEFAULT_END_DATE = datetime(2025, 7, 26)
DEFAULT_FORECAST_HOUR = 120


@dataclass(frozen=True)
class ConversionPlan:
    base_time: datetime
    valid_time: datetime
    source_npy: Path
    target_dir: Path
    target_nc: Path


@dataclass(frozen=True)
class ConversionResult:
    plan: ConversionPlan
    status: str
    message: str


def time_to_str(value: datetime) -> str:
    return value.strftime("%Y-%m-%d-%H-%M")


def parse_date(value: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y-%m-%d-%H-%M", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"Unsupported date {value!r}; use YYYY-MM-DD, YYYY-MM-DD-HH-MM, or YYYYMMDD."
    )


def iter_daily_base_times(start_date: datetime, end_date: datetime) -> list[datetime]:
    if end_date < start_date:
        raise ValueError("end_date must be greater than or equal to start_date")

    base_times = []
    current = start_date
    while current <= end_date:
        base_times.append(current)
        current += timedelta(days=1)
    return base_times


def build_conversion_plan(
    output_root: Path,
    start_date: datetime,
    end_date: datetime,
    forecast_hour: int = DEFAULT_FORECAST_HOUR,
) -> list[ConversionPlan]:
    plans = []
    for base_time in iter_daily_base_times(start_date, end_date):
        valid_time = base_time + timedelta(hours=forecast_hour)
        base_dir = output_root / time_to_str(base_time)
        source_npy = base_dir / "timeline_cache" / f"output_surface_{time_to_str(valid_time)}.npy"
        target_dir = base_dir / str(forecast_hour)
        target_nc = target_dir / f"output_surface_{time_to_str(valid_time)}.nc"
        plans.append(
            ConversionPlan(
                base_time=base_time,
                valid_time=valid_time,
                source_npy=source_npy,
                target_dir=target_dir,
                target_nc=target_nc,
            )
        )
    return plans


def decode_surface_cache_to_nc(
    plan: ConversionPlan,
    surface_decoder: Callable[[str, str, str], None],
    overwrite: bool = False,
) -> ConversionResult:
    if not plan.source_npy.exists():
        return ConversionResult(plan, "missing_source", f"Missing source: {plan.source_npy}")

    if plan.target_nc.exists() and not overwrite:
        return ConversionResult(plan, "skipped_existing", f"Existing target: {plan.target_nc}")

    plan.target_dir.mkdir(parents=True, exist_ok=True)
    if plan.target_nc.exists() and overwrite:
        plan.target_nc.unlink()

    surface_decoder(str(plan.source_npy), plan.target_nc.name, str(plan.target_dir))
    if not plan.target_nc.exists():
        return ConversionResult(plan, "failed", f"Decoder did not create: {plan.target_nc}")

    return ConversionResult(plan, "created", f"Created: {plan.target_nc}")


def run_conversion(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    start_date: datetime = DEFAULT_START_DATE,
    end_date: datetime = DEFAULT_END_DATE,
    forecast_hour: int = DEFAULT_FORECAST_HOUR,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[ConversionResult]:
    plans = build_conversion_plan(output_root, start_date, end_date, forecast_hour)
    if dry_run:
        return [
            ConversionResult(
                plan,
                "dry_run",
                f"{plan.source_npy} -> {plan.target_nc}",
            )
            for plan in plans
        ]

    from forecast_decode_functions import surface

    return [
        decode_surface_cache_to_nc(plan, surface_decoder=surface, overwrite=overwrite)
        for plan in plans
    ]


def summarize_results(results: list[ConversionResult]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode ERA5 timeline_cache 120h surface .npy files into sibling 120 folders."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", type=parse_date, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=parse_date, default=DEFAULT_END_DATE)
    parser.add_argument("--forecast-hour", type=int, default=DEFAULT_FORECAST_HOUR)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target nc files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned conversions without writing.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_conversion(
        output_root=args.output_root,
        start_date=args.start_date,
        end_date=args.end_date,
        forecast_hour=args.forecast_hour,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    for result in results:
        base = time_to_str(result.plan.base_time)
        valid = time_to_str(result.plan.valid_time)
        print(f"[{result.status}] base={base} valid={valid} {result.message}")

    summary = summarize_results(results)
    summary_text = ", ".join(f"{key}={value}" for key, value in sorted(summary.items()))
    print(f"Summary: {summary_text}")

    if summary.get("missing_source") or summary.get("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
