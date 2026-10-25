"""Run four wind forecast experiments for daily T0 values (UTC)."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import infer as inference
from experiments import DEFAULT_LEAD_HOURS, EXPERIMENTS, PROJECT_ROOT, time_label


def parse_date(text):
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use YYYY-MM-DD") from exc


def parse_leads(text):
    try:
        return inference.normalize_leads(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated nonnegative integer lead hours") from exc


def build_parser():
    parser = argparse.ArgumentParser(description="Four Pangu experiments; save compressed u10/v10 and delete successful task caches.")
    parser.add_argument("--start-date", type=parse_date, default=datetime(2025, 1, 1), help="First T0 date, inclusive (UTC)")
    parser.add_argument("--end-date", type=parse_date, default=datetime(2025, 12, 31), help="Last T0 date, inclusive (UTC)")
    parser.add_argument("--hour", type=int, choices=range(24), default=0, help="Daily T0 hour in UTC")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--lead-hours", type=parse_leads, default=list(DEFAULT_LEAD_HOURS), help="Default: 0,3,...,72; 0 is always included")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None, help="Default: <project-root>/model_output/experiments")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild selected tasks and their temporary caches")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of CUDA")
    parser.add_argument("--dry-run", action="store_true", help="Check required input/model paths without inference or writes")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.end_date < args.start_date:
        raise ValueError("end-date must be on or after start-date")
    args.project_root = args.project_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else args.project_root / "model_output" / "experiments"
    names = list(dict.fromkeys(args.experiments))
    targets = []
    current = args.start_date.replace(hour=args.hour)
    while current.date() <= args.end_date.date():
        targets.append(current)
        current += timedelta(days=1)
    inputs = sorted({inference.input_directory(args.project_root, EXPERIMENTS[name], target)
                     for target in targets for name in names})
    required = [path / file for path in inputs for file in ("input_surface.npy", "input_upper.npy")]
    steps = {step for name in names for lead in args.lead_hours
             for _, step in inference.integration_steps(EXPERIMENTS[name].model_hour(lead))}
    required.extend(args.project_root / "models" / f"pangu_weather_{step}.onnx" for step in sorted(steps))
    missing = [path for path in required if not path.is_file()]
    print(f"T0: {time_label(targets[0])} .. {time_label(targets[-1])} UTC")
    print(f"试验: {', '.join(names)}; 时效: {args.lead_hours}")
    print(f"任务数: {len(targets) * len(names)}; 共享输入时次数: {len(inputs)}")
    print(f"输出根目录: {output_root}; 缺失文件数: {len(missing)}")
    for path in missing[:10]:
        print(f"[缺失] {path}")
    if args.dry_run:
        return 1 if missing else 0

    inference.use_GPU = not args.cpu
    report = {"start_date": args.start_date.isoformat(), "end_date": args.end_date.isoformat(),
              "hour_utc": args.hour, "experiments": names, "lead_hours": args.lead_hours, "tasks": []}
    report_path = output_root / "_runs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    failed = 0
    for target in targets:
        for name in names:
            row = {"experiment": name, "target_time_utc": target.isoformat()}
            try:
                outputs = inference.infer_experiment(target, name, args.lead_hours, project_root=args.project_root,
                                                     output_root=output_root, overwrite=args.overwrite)
                row.update(status="complete", outputs=[str(path) for path in outputs])
            except Exception as exc:
                failed += 1
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                print(f"[失败] {name} T0={time_label(target)}: {exc}")
            report["tasks"].append(row)
            inference.atomic_json(report_path, report)
    print(f"任务完成: {len(report['tasks']) - failed}; 失败: {failed}; 记录: {report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
