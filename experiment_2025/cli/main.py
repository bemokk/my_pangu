"""Command-line entry point for the 2025 four-experiment inference."""

from __future__ import annotations

import argparse
import json
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiment_2025"

from ..core.config import DEFAULT_LEADS, ERA5_ROOT, GDAS_ROOT, MODEL_ROOT, OUTPUT_ROOT
from ..core.experiments import EXPERIMENTS, time_label
from ..core.inference import infer_experiment, integration_steps, normalize_leads
from ..core.input_data import Era5Archive
from ..gdas.paths import input_directory as gdas_input_directory


def _date(text: str) -> datetime:
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期必须为 YYYY-MM-DD") from exc


def _leads(text: str) -> list[int]:
    try:
        return normalize_leads(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _targets(start: datetime, end: datetime, hour: int) -> list[datetime]:
    if end < start:
        raise ValueError("结束日期不能早于开始日期")
    current = start.replace(hour=hour)
    result = []
    while current.date() <= end.date():
        result.append(current)
        current += timedelta(days=1)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2025 Pangu four-experiment inference")
    parser.add_argument("--start-date", type=_date, default=datetime(2025, 1, 1))
    parser.add_argument("--end-date", type=_date, default=datetime(2025, 12, 31))
    parser.add_argument("--hour", type=int, choices=range(24), default=0, help="T0 UTC hour")
    parser.add_argument("--experiments", nargs="+", choices=tuple(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--lead-hours", type=_leads, default=list(DEFAULT_LEADS))
    parser.add_argument("--era5-root", type=Path, default=ERA5_ROOT)
    parser.add_argument("--gdas-root", type=Path, default=GDAS_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of CUDA")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Check coverage, models and paths only")
    return parser


def _dry_run(args, targets, experiments) -> int:
    missing = []
    required_gdas = {
        experiment.initial_time(target)
        for experiment in experiments if experiment.source == "gdas"
        for target in targets
    }
    for initial_time in sorted(required_gdas):
        directory = gdas_input_directory(args.gdas_root, initial_time)
        for name in ("input_surface.npy", "input_upper.npy"):
            path = directory / name
            if not path.is_file():
                missing.append(str(path))
    for step in sorted({step for experiment in experiments for lead in args.lead_hours for _, step in integration_steps(experiment.model_hour(lead))}):
        path = args.model_root / f"pangu_weather_{step}.onnx"
        if not path.exists():
            missing.append(str(path))
    print(f"T0: {time_label(targets[0])} .. {time_label(targets[-1])}")
    print(f"试验: {', '.join(experiment.name for experiment in experiments)}")
    print(f"时效: {args.lead_hours}; 任务数: {len(targets) * len(experiments)}")
    print(f"ERA5根目录: {args.era5_root}")
    print(f"GDAS根目录: {args.gdas_root}")
    print(f"输出根目录: {args.output_root}")
    required_era5 = {
        experiment.initial_time(target)
        for experiment in experiments if experiment.source == "era5"
        for target in targets
    }
    archive = Era5Archive(args.era5_root)
    for year in sorted({stamp.year for stamp in required_era5}):
        try:
            available = archive.available_times(year)
        except Exception as exc:
            missing.append(f"ERA5 {year}: {exc}")
            continue
        absent = sorted(stamp for stamp in required_era5 if stamp.year == year and stamp not in available)
        if absent:
            missing.append(f"ERA5 missing {len(absent)} initial times in {year}: {absent[:3]}")
        print(f"ERA5 {year}: {len(available)} paired surface/upper times")
    if missing:
        print(f"缺失/错误共 {len(missing)} 项（最多显示20项）:")
        for item in missing[:20]:
            print(f"  {item}")
    else:
        print("检查通过：输入覆盖和模型文件齐全")
    return 1 if missing else 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    targets = _targets(args.start_date, args.end_date, args.hour)
    experiments = [EXPERIMENTS[name] for name in dict.fromkeys(args.experiments)]
    if args.dry_run:
        return _dry_run(args, targets, experiments)

    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "_runs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    report = {
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "hour_utc": args.hour,
        "experiments": [experiment.name for experiment in experiments],
        "lead_hours": args.lead_hours,
        "tasks": [],
    }
    failed = 0
    batch_started = time.perf_counter()
    total_tasks = len(targets) * len(experiments)
    print(f"[批次开始] {args.start_date:%Y-%m-%d} .. {args.end_date:%Y-%m-%d} "
          f"任务={total_tasks} 每日时效={len(args.lead_hours)}，计时报告={report_path}", flush=True)
    for target in targets:
        for experiment in experiments:
            row = {"experiment": experiment.name, "target_time_utc": target.isoformat()}
            task_started = time.perf_counter()
            task_index = len(report["tasks"]) + 1
            timing = {}
            stop_heartbeat = threading.Event()

            def heartbeat():
                while not stop_heartbeat.wait(30):
                    print(f"[运行中] {task_index}/{total_tasks} {experiment.name} {time_label(target)} "
                          f"本日用时={time.perf_counter()-task_started:.0f}s "
                          f"批次用时={time.perf_counter()-batch_started:.0f}s", flush=True)

            heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
            heartbeat_thread.start()
            print(f"[任务开始] {task_index}/{total_tasks} {experiment.name} {time_label(target)} UTC", flush=True)
            try:
                outputs = infer_experiment(
                    target,
                    experiment.name,
                    args.lead_hours,
                    era5_root=args.era5_root,
                    gdas_root=args.gdas_root,
                    output_root=args.output_root,
                    model_root=args.model_root,
                    overwrite=args.overwrite,
                    use_gpu=not args.cpu,
                    timings=timing,
                )
                row.update(status="complete", outputs=[str(path) for path in outputs])
            except Exception as exc:  # keep the annual batch moving
                failed += 1
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                print(f"[失败] {experiment.name} {time_label(target)}: {exc}")
            finally:
                stop_heartbeat.set()
                heartbeat_thread.join()
                row["elapsed_seconds"] = time.perf_counter() - task_started
                row["timings"] = timing
            report["tasks"].append(row)
            report["elapsed_seconds"] = time.perf_counter() - batch_started
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = report_path.with_suffix(".json.part")
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(report_path)
            print(f"[批次进度] {len(report['tasks'])}/{total_tasks} 完成={len(report['tasks'])-failed} "
                  f"失败={failed} 累计用时={report['elapsed_seconds']:.1f}s", flush=True)
    print(f"任务完成={len(report['tasks']) - failed}, 失败={failed}, 报告={report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())



