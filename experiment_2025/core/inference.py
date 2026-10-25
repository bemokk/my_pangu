"""Four-experiment Pangu inference for the 2025 archive."""

from __future__ import annotations

import gc
import json
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

from .config import (
    DEFAULT_LEADS,
    GRID_SHAPE,
    OUTPUT_ROOT,
    STATE_CACHE_VERSION,
    model_path,
)
from .experiments import EXPERIMENTS, Experiment, run_directory, time_label, wind_path
from .input_data import InitialState, source_for_experiment
from .wind_output import validate_wind_file, write_wind_file


USE_GPU = True


def normalize_leads(values) -> list[int]:
    values = list(values)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0 for value in values):
        raise ValueError("Lead hours must be nonnegative integers")
    return sorted({0, *(int(value) for value in values)})


def integration_steps(model_hour: int):
    """Return the legacy greedy 24/6/3/1-hour model sequence."""
    if model_hour < 0:
        raise ValueError("model_hour must be nonnegative")
    current = 0
    while current < model_hour:
        step = next(size for size in (24, 6, 3, 1) if size <= model_hour - current)
        current += step
        yield current, step


def _endpoint_order(experiment: Experiment, lead: int) -> tuple[int, int]:
    hour = experiment.model_hour(lead)
    if hour == 0:
        return (0, 0)
    last_step = list(integration_steps(hour))[-1][1]
    return ((24, 6, 3, 1).index(last_step) + 1, hour)


def _file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def _cache_paths(cache_dir: Path, hour: int) -> tuple[Path, Path, Path]:
    return (
        cache_dir / f"upper_{hour:04d}h.npy",
        cache_dir / f"surface_{hour:04d}h.npy",
        cache_dir / f"state_{hour:04d}h.json",
    )


def _validate_state(state: tuple[np.ndarray, np.ndarray]):
    upper, surface = state[:2]
    if upper.shape != (5, 13, *GRID_SHAPE) or surface.shape != (4, *GRID_SHAPE):
        raise ValueError(f"Expected upper=(5,13,721,1440), surface=(4,721,1440); got {upper.shape}, {surface.shape}")
    if upper.dtype != np.float32 or surface.dtype != np.float32:
        raise ValueError(f"Pangu states must be float32; got {upper.dtype}, {surface.dtype}")
    if any(not np.isfinite(channel).all() for channel in upper) or not np.isfinite(surface).all():
        raise ValueError("Pangu state contains non-finite values")


def _save_state(cache_dir: Path, hour: int, upper: np.ndarray, surface: np.ndarray) -> None:
    _validate_state((upper, surface))
    upper_path, surface_path, marker = _cache_paths(cache_dir, hour)
    if marker.exists():
        marker.unlink()
    _atomic_npy(upper_path, upper)
    _atomic_npy(surface_path, surface)
    _atomic_json(marker, [_file_identity(upper_path), _file_identity(surface_path)])


def _load_state(cache_dir: Path, hour: int) -> tuple[np.ndarray, np.ndarray] | None:
    upper_path, surface_path, marker = _cache_paths(cache_dir, hour)
    if not marker.exists() or not upper_path.exists() or not surface_path.exists():
        return None
    try:
        if json.loads(marker.read_text(encoding="utf-8")) != [_file_identity(upper_path), _file_identity(surface_path)]:
            return None
        upper = np.load(upper_path, mmap_mode="r", allow_pickle=False)
        surface = np.load(surface_path, mmap_mode="r", allow_pickle=False)
        _validate_state((upper, surface))
        return upper, surface
    except (OSError, ValueError, EOFError):
        return None


def _create_ort_session(model: Path, use_gpu: bool = USE_GPU):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.intra_op_num_threads = 30
    if use_gpu:
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch cannot access CUDA")
        except ImportError as exc:
            raise RuntimeError("GPU inference requires a CUDA-enabled PyTorch installation") from exc
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("onnxruntime has no CUDAExecutionProvider")
        providers = [("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    if use_gpu and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDAExecutionProvider was not enabled")
    return session


def _remove_cache(cache_dir: Path, run_dir: Path) -> None:
    if not cache_dir.exists():
        return
    if cache_dir.is_symlink() or cache_dir.resolve() != run_dir.resolve() / "timeline_cache":
        raise RuntimeError(f"Refusing to remove unexpected cache path: {cache_dir}")
    shutil.rmtree(cache_dir)


def _metadata(experiment: Experiment, target_time: datetime, lead: int, source_files: tuple[Path, ...], model_ids: dict) -> dict:
    return {
        "experiment": experiment.name,
        "experiment_label": experiment.label,
        "source": experiment.source,
        "target_time_utc": target_time.isoformat(),
        "initial_time_utc": experiment.initial_time(target_time).isoformat(),
        "valid_time_utc": (target_time + timedelta(hours=lead)).isoformat(),
        "lead_hour": int(lead),
        "model_forecast_hour": int(experiment.model_hour(lead)),
        "lag_hours": int(experiment.lag_hours),
        "source_files": json.dumps([str(path) for path in source_files], ensure_ascii=False),
        "input_identity": json.dumps([_file_identity(path) for path in source_files], sort_keys=True),
        "model_files": json.dumps(model_ids, sort_keys=True),
        "format_version": int(STATE_CACHE_VERSION),
    }


def _load_initial(source, initial_time: datetime) -> InitialState:
    state = source.load(initial_time)
    _validate_state((state.upper, state.surface))
    return state


def infer_experiment(
    target_time: datetime,
    experiment_name: str,
    lead_hours=DEFAULT_LEADS,
    *,
    era5_root: Path | None = None,
    gdas_root: Path | None = None,
    output_root: Path = OUTPUT_ROOT,
    model_root: Path | None = None,
    overwrite: bool = False,
    use_gpu: bool = USE_GPU,
    session_factory: Callable | None = None,
    timings: dict | None = None,
) -> list[Path]:
    """Run one experiment at one T0, retaining only compressed u10/v10 outputs."""
    if target_time.tzinfo is not None or target_time.minute or target_time.second or target_time.microsecond:
        raise ValueError("target_time must be a naive UTC whole-hour datetime")
    if experiment_name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment {experiment_name!r}; choose from {tuple(EXPERIMENTS)}")
    experiment = EXPERIMENTS[experiment_name]
    run_started = time.perf_counter()
    timings = timings if timings is not None else {}
    timings.update(input_seconds=0.0, model_load_seconds=0.0, inference_seconds=0.0,
                   cache_seconds=0.0, output_seconds=0.0, model_steps=0, outputs_written=0)
    leads = normalize_leads(lead_hours)
    source = source_for_experiment(experiment, era5_root=era5_root, gdas_root=gdas_root)
    initial_time = experiment.initial_time(target_time)
    print(f"[读取] {experiment.name} {time_label(target_time)} 初始资料={initial_time.isoformat()} UTC", flush=True)
    phase_started = time.perf_counter()
    initial = _load_initial(source, initial_time)
    timings["input_seconds"] = time.perf_counter() - phase_started
    print(f"[读取完成] {timings['input_seconds']:.2f}s", flush=True)
    output_root = Path(output_root)
    run_dir = run_directory(output_root, experiment, target_time)
    cache_dir = run_dir / "timeline_cache"
    if cache_dir.is_symlink() or cache_dir.resolve() != run_dir.resolve() / "timeline_cache":
        raise RuntimeError(f"Unexpected cache path: {cache_dir}")
    model_root = Path(model_root) if model_root is not None else model_path(1).parents[0]
    steps_needed = sorted({step for lead in leads for _, step in integration_steps(experiment.model_hour(lead))})
    models = {step: model_root / f"pangu_weather_{step}.onnx" for step in steps_needed}
    missing_models = [path for path in models.values() if not path.exists()]
    if missing_models:
        raise FileNotFoundError("Missing Pangu models:\n" + "\n".join(f"  {path}" for path in missing_models))
    model_ids = {str(step): _file_identity(path) for step, path in models.items()}
    metadata = {lead: _metadata(experiment, target_time, lead, initial.source_files, model_ids) for lead in leads}
    output_paths = {lead: wind_path(output_root, experiment, target_time, lead) for lead in leads}
    missing_leads = [lead for lead in leads if overwrite or not validate_wind_file(output_paths[lead], initial.latitude, initial.longitude, metadata[lead])]
    if not missing_leads:
        _remove_cache(cache_dir, run_dir)
        timings["total_seconds"] = time.perf_counter() - run_started
        timings["outputs_reused"] = len(leads)
        print(f"[跳过] 已验证 {len(leads)} 个时效，总耗时={timings['total_seconds']:.2f}s", flush=True)
        return [output_paths[lead] for lead in leads]
    timings["outputs_reused"] = len(leads) - len(missing_leads)

    manifest = {
        "version": STATE_CACHE_VERSION,
        "experiment": experiment.name,
        "target_time_utc": target_time.isoformat(),
        "initial_time_utc": initial_time.isoformat(),
        "source_files": [_file_identity(path) for path in initial.source_files],
        "models": model_ids,
        "grid_shape": list(GRID_SHAPE),
    }
    manifest_path = cache_dir / "manifest.json"
    if cache_dir.exists():
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old_manifest = None
        if overwrite:
            _remove_cache(cache_dir, run_dir)
        elif old_manifest != manifest:
            raise RuntimeError(f"Cache identity changed at {cache_dir}; use --overwrite")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(manifest_path, manifest)

    session_factory = session_factory or (lambda model: _create_ort_session(model, use_gpu))
    session = None
    session_step = None
    upper = surface = cached = None
    try:
        for lead in sorted(missing_leads, key=lambda value: _endpoint_order(experiment, value)):
            model_hour = experiment.model_hour(lead)
            upper = surface = cached = None
            for next_hour, step in integration_steps(model_hour):
                cached = _load_state(cache_dir, next_hour)
                if cached is not None:
                    upper, surface = cached
                    continue
                if next_hour == step:
                    upper = initial.upper
                    surface = initial.surface
                if session_step != step:
                    session = None
                    phase_started = time.perf_counter()
                    print(f"[加载模型] {step}h", flush=True)
                    session = session_factory(models[step])
                    duration = time.perf_counter() - phase_started
                    timings["model_load_seconds"] += duration
                    print(f"[模型已加载] {step}h 耗时={duration:.2f}s", flush=True)
                    session_step = step
                print(f"[推理] {experiment.name} {time_label(target_time)} +{next_hour}h ({step}h model)")
                phase_started = time.perf_counter()
                upper, surface = session.run(None, {"input": np.asarray(upper), "input_surface": np.asarray(surface)})
                duration = time.perf_counter() - phase_started
                timings["inference_seconds"] += duration
                timings["model_steps"] += 1
                print(f"[积分完成] +{next_hour}h 耗时={duration:.2f}s", flush=True)
                upper = np.asarray(upper, dtype=np.float32)
                surface = np.asarray(surface, dtype=np.float32)
                phase_started = time.perf_counter()
                _save_state(cache_dir, next_hour, upper, surface)
                timings["cache_seconds"] += time.perf_counter() - phase_started
            if model_hour == 0:
                upper, surface = initial.upper, initial.surface
            phase_started = time.perf_counter()
            write_wind_file(output_paths[lead], surface, initial.latitude, initial.longitude, metadata[lead])
            timings["output_seconds"] += time.perf_counter() - phase_started
            timings["outputs_written"] += 1
            print(f"[风场已保存] +{lead}h 已完成={timings['outputs_reused'] + timings['outputs_written']}/{len(leads)} "
                  f"本日已用时={time.perf_counter()-run_started:.1f}s", flush=True)
            upper = surface = cached = None
        if not all(validate_wind_file(output_paths[lead], initial.latitude, initial.longitude, metadata[lead]) for lead in leads):
            raise RuntimeError("At least one requested wind output failed validation; cache retained")
    finally:
        session = upper = surface = cached = None
        gc.collect()
    _remove_cache(cache_dir, run_dir)
    timings["total_seconds"] = time.perf_counter() - run_started
    print(f"[本日完成] {experiment.name} {time_label(target_time)} {len(leads)}个时效 "
          f"总耗时={timings['total_seconds']:.2f}s，缓存已清理", flush=True)
    return [output_paths[lead] for lead in leads]
