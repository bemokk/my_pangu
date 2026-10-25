"""Check Python dependencies, model files, and ONNX Runtime providers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SIZE = 1_181_711_187
MODEL_HASHES = {
    "pangu_weather_1.onnx": "179E5029C453AE459DFD52F14610A6C5F5AD39F1371985744EA0CE6C546FDA2A",
    "pangu_weather_3.onnx": "57533942F1C8B3F02B8D9AC26E89852EBDF9682CDC66EDD7402F536C0778699A",
    "pangu_weather_6.onnx": "8A4C022A7DD65FDC235757BA907574BC9A1FC0223A2EA4A4800961A52F475E97",
    "pangu_weather_24.onnx": "613A5C140A1399ABCAFFB4DBCE32AF373A1F5F56C515704F5BE61925BB9FDCFD",
}
MODULES = ["cdsapi", "netCDF4", "numpy", "pygrib", "requests", "tqdm", "onnxruntime"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--hash-models", action="store_true", help="also perform the slower SHA-256 check")
    args = parser.parse_args()
    failures: list[str] = []

    print(f"Python: {sys.version.split()[0]}")
    for module in MODULES:
        found = importlib.util.find_spec(module) is not None
        print(f"dependency {module}: {'OK' if found else 'MISSING'}")
        if not found:
            failures.append(f"dependency: {module}")

    ort = None
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print("ONNX Runtime providers:", ", ".join(providers))
        if args.device == "gpu" and "CUDAExecutionProvider" not in providers:
            failures.append("CUDAExecutionProvider")
    except ImportError:
        pass

    for name, expected_hash in MODEL_HASHES.items():
        path = PROJECT_ROOT / "models" / name
        if not path.exists():
            print(f"model {name}: MISSING")
            failures.append(f"model: {name}")
            continue
        size_ok = path.stat().st_size == MODEL_SIZE
        print(f"model {name}: {'size OK' if size_ok else 'wrong size'}")
        if not size_ok:
            failures.append(f"model size: {name}")
        if args.hash_models:
            hash_ok = sha256(path) == expected_hash
            print(f"model {name}: {'hash OK' if hash_ok else 'wrong hash'}")
            if not hash_ok:
                failures.append(f"model hash: {name}")

    test_model = PROJECT_ROOT / "models" / "pangu_weather_3.onnx"
    if args.device == "gpu" and ort is not None and test_model.exists():
        try:
            import torch

            print(
                "PyTorch CUDA:",
                torch.version.cuda,
                "| available:",
                torch.cuda.is_available(),
            )
            if not torch.cuda.is_available():
                failures.append("PyTorch CUDA runtime")
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()

            print("Testing a real CUDA ONNX session (this can take about one minute)...")
            session = ort.InferenceSession(
                str(test_model),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            actual_providers = session.get_providers()
            print("Test session providers:", ", ".join(actual_providers))
            if "CUDAExecutionProvider" not in actual_providers:
                failures.append("real CUDA ONNX session")
            del session
        except Exception as exc:
            print(f"CUDA session test failed: {exc}")
            failures.append("real CUDA ONNX session")

    if failures:
        print("\nSetup is incomplete:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nSetup looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
