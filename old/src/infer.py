"""Pangu batch inference with temporary full states and permanent 10-m winds."""

import json
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np

from experiments import DEFAULT_LEAD_HOURS, EXPERIMENTS, PROJECT_ROOT, run_directory, time_label, wind_path
from wind_output import input_grid, validate_wind_file, write_wind_file


use_GPU = True
GRID_SHAPE = (721, 1440)
CACHE_VERSION = 1


def create_ort_session(model_path, options=None):
    # Load CUDA DLLs through torch before importing ONNX Runtime on Windows.
    import torch
    import onnxruntime as ort

    if options is None:
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.intra_op_num_threads = 30
    if use_GPU:
        if not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA unavailable; install/configure torch and onnxruntime-gpu, or use --cpu")
        providers = [("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    if use_GPU and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("ONNX Runtime failed to enable CUDAExecutionProvider")
    return session


def normalize_leads(lead_hours):
    values = list(lead_hours)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0 for value in values):
        raise ValueError("Lead hours must be nonnegative integers")
    return sorted({0, *(int(value) for value in values)})


def integration_steps(model_hour):
    """Preserve the original greedy 24/6/3/1-h integration for each endpoint."""
    if model_hour < 0:
        raise ValueError("Model hour cannot be negative")
    current = 0
    while current < model_hour:
        step = next(size for size in (24, 6, 3, 1) if size <= model_hour - current)
        current += step
        yield current, step


def endpoint_order(experiment, lead):
    hour = experiment.model_hour(lead)
    if hour == 0:
        return (0, 0)
    last_step = next(reversed(list(integration_steps(hour))))[1]
    return ((24, 6, 3, 1).index(last_step) + 1, hour)


def input_directory(project_root, experiment, target_time):
    return Path(project_root) / "model_input" / "single_time_point" / experiment.source / time_label(experiment.initial_time(target_time))


def file_identity(path):
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_npy(path, values):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    temporary.replace(path)


def output_metadata(experiment, target_time, lead, input_identity, model_identity):
    return {
        "experiment": experiment.name,
        "source": experiment.source,
        "initial_time_utc": experiment.initial_time(target_time).isoformat(),
        "target_time_utc": target_time.isoformat(),
        "valid_time_utc": (target_time + timedelta(hours=lead)).isoformat(),
        "lead_hour": int(lead),
        "model_forecast_hour": experiment.model_hour(lead),
        "lag_hours": experiment.lag_hours,
        "input_identity": json.dumps(input_identity, sort_keys=True),
        "model_identity": json.dumps(model_identity, sort_keys=True),
        "format_version": CACHE_VERSION,
    }


def validate_state(upper, surface):
    if upper.shape != (5, 13, *GRID_SHAPE) or surface.shape != (4, *GRID_SHAPE):
        raise ValueError(f"Invalid Pangu state shapes: upper={upper.shape}, surface={surface.shape}")
    if upper.dtype != np.float32 or surface.dtype != np.float32:
        raise ValueError("Pangu states must be float32")
    if any(not np.isfinite(channel).all() for channel in upper) or not np.isfinite(surface).all():
        raise ValueError("Pangu state contains non-finite values")


def cache_paths(cache_dir, hour):
    return cache_dir / f"upper_{hour}.npy", cache_dir / f"surface_{hour}.npy", cache_dir / f"state_{hour}.json"


def load_cached_state(cache_dir, hour):
    upper_path, surface_path, marker = cache_paths(cache_dir, hour)
    if not marker.exists():
        return None
    try:
        # Marker is committed last; detect interrupted rewrites of either array.
        expected = json.loads(marker.read_text(encoding="utf-8"))
        if expected != [file_identity(upper_path), file_identity(surface_path)]:
            return None
        upper = np.load(upper_path, mmap_mode="r", allow_pickle=False)
        surface = np.load(surface_path, mmap_mode="r", allow_pickle=False)
        validate_state(upper, surface)
        return upper, surface
    except (OSError, ValueError, EOFError):
        return None


def save_cached_state(cache_dir, hour, upper, surface):
    validate_state(upper, surface)
    upper_path, surface_path, marker = cache_paths(cache_dir, hour)
    if marker.exists():
        marker.unlink()
    atomic_npy(upper_path, upper)
    atomic_npy(surface_path, surface)
    atomic_json(marker, [file_identity(upper_path), file_identity(surface_path)])


def remove_task_cache(cache_dir, run_dir):
    if not cache_dir.exists():
        return
    if cache_dir.is_symlink() or cache_dir.resolve() != run_dir.resolve() / "timeline_cache":
        raise RuntimeError(f"Unsafe cache location: {cache_dir}")
    shutil.rmtree(cache_dir)


def infer_experiment(target_time, experiment_name, lead_hours=DEFAULT_LEAD_HOURS, *,
                     project_root=PROJECT_ROOT, output_root=None, overwrite=False, session_factory=None):
    """Complete one T0/experiment; remove cache only after all outputs validate.

    Failures retain full state checkpoints for resumption. Existing July outputs
    are never read, overwritten, or removed by this function.
    """
    if target_time.tzinfo is not None or any((target_time.minute, target_time.second, target_time.microsecond)):
        raise ValueError("T0 must be a whole hour expressed as naive UTC")
    experiment = EXPERIMENTS[experiment_name]
    leads = normalize_leads(lead_hours)
    project_root = Path(project_root)
    output_root = Path(output_root) if output_root is not None else project_root / "model_output" / "experiments"
    run_dir = run_directory(output_root, experiment, target_time)
    cache_dir = run_dir / "timeline_cache"
    if cache_dir.is_symlink() or cache_dir.resolve() != run_dir.resolve() / "timeline_cache":
        raise RuntimeError(f"Unsafe cache location: {cache_dir}")
    source_dir = input_directory(project_root, experiment, target_time)
    surface_path = source_dir / "input_surface.npy"
    upper_path = source_dir / "input_upper.npy"
    inputs = [file_identity(path) for path in (upper_path, surface_path)]
    if (source_dir / "surface.nc").exists():
        inputs.append(file_identity(source_dir / "surface.nc"))
    steps_needed = sorted({step for lead in leads for _, step in integration_steps(experiment.model_hour(lead))})
    models = {step: project_root / "models" / f"pangu_weather_{step}.onnx" for step in steps_needed}
    model_ids = {str(step): file_identity(path) for step, path in models.items()}
    latitude, longitude = input_grid(source_dir, GRID_SHAPE)
    metadata = {lead: output_metadata(experiment, target_time, lead, inputs, model_ids) for lead in leads}
    paths = {lead: wind_path(output_root, experiment, target_time, lead) for lead in leads}
    missing = [lead for lead in leads if overwrite or not validate_wind_file(paths[lead], latitude, longitude, metadata[lead])]
    if not missing:
        remove_task_cache(cache_dir, run_dir)
        print(f"[完成，跳过] {experiment.name} T0={time_label(target_time)}")
        return list(paths.values())

    identity = {"version": CACHE_VERSION, "experiment": experiment.name, "target_time": target_time.isoformat(),
                "inputs": inputs, "models": model_ids, "grid_shape": list(GRID_SHAPE)}
    manifest_path = cache_dir / "manifest.json"
    if cache_dir.exists():
        try:
            cached_identity = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached_identity = None
        if overwrite:
            remove_task_cache(cache_dir, run_dir)
        elif cached_identity != identity:
            raise RuntimeError(f"Cache configuration/input changed: {cache_dir}; use --overwrite to rebuild this task")
    cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_path, identity)
    session_factory = session_factory or create_ort_session
    session = None
    session_step = None
    upper = surface = cached = None
    try:
        # Compute 24h endpoints first, then 6h and 3h endpoints. For the default
        # schedule this loads each ONNX model once while preserving every path.
        for lead in sorted(missing, key=lambda value: endpoint_order(experiment, value)):
            hour = experiment.model_hour(lead)
            current_hour = 0
            upper = surface = cached = None
            for next_hour, step in integration_steps(hour):
                cached = load_cached_state(cache_dir, next_hour)
                if cached is not None:
                    upper, surface = cached
                else:
                    if current_hour == 0:
                        upper = np.load(upper_path, allow_pickle=False)
                        surface = np.load(surface_path, allow_pickle=False)
                        validate_state(upper, surface)
                    if session_step != step:
                        session = None
                        session = session_factory(models[step])
                        session_step = step
                    print(f"[推理] {experiment.name} T0={time_label(target_time)} model +{current_hour} -> +{next_hour}h")
                    upper, surface = session.run(None, {"input": np.asarray(upper), "input_surface": np.asarray(surface)})
                    save_cached_state(cache_dir, next_hour, upper, surface)
                current_hour = next_hour
            if hour == 0:
                surface = np.load(surface_path, allow_pickle=False)
                upper = np.load(upper_path, allow_pickle=False)
                validate_state(upper, surface)
            write_wind_file(paths[lead], surface, latitude, longitude, metadata[lead])
            upper = surface = cached = None
        if not all(validate_wind_file(paths[lead], latitude, longitude, metadata[lead]) for lead in leads):
            raise RuntimeError("Some requested wind files failed validation; preserving timeline_cache")
    finally:
        # Release mmap references before removing checkpoint files on Windows.
        upper = surface = cached = session = None
    remove_task_cache(cache_dir, run_dir)
    print(f"[完成，缓存已删除] {experiment.name} T0={time_label(target_time)}")
    return list(paths.values())


def infer(start_time, forecast_hour, dataType):
    """Single-lead entry: gdas/era5 mean direct forecasts at T0."""
    name = {"gdas": "gdas_direct", "era5": "era5_direct"}.get(dataType.lower(), dataType.lower())
    return infer_experiment(start_time, name, [forecast_hour])
