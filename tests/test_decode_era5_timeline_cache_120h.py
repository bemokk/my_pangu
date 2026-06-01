from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import decode_era5_timeline_cache_120h as decoder  # noqa: E402


def test_build_conversion_plan_points_120h_cache_to_120_folder(tmp_path):
    root = tmp_path / "model_output" / "era5"
    base_dir = root / "2025-06-26-00-00"
    cache_dir = base_dir / "timeline_cache"
    cache_dir.mkdir(parents=True)
    source = cache_dir / "output_surface_2025-07-01-00-00.npy"
    source.write_bytes(b"fake")

    plans = decoder.build_conversion_plan(
        output_root=root,
        start_date=datetime(2025, 6, 26),
        end_date=datetime(2025, 6, 26),
        forecast_hour=120,
    )

    assert plans == [
        decoder.ConversionPlan(
            base_time=datetime(2025, 6, 26),
            valid_time=datetime(2025, 7, 1),
            source_npy=source,
            target_dir=base_dir / "120",
            target_nc=base_dir / "120" / "output_surface_2025-07-01-00-00.nc",
        )
    ]


def test_decode_surface_cache_to_nc_uses_supplied_decoder(tmp_path):
    source = tmp_path / "timeline_cache" / "output_surface_2025-07-01-00-00.npy"
    source.parent.mkdir()
    source.write_bytes(b"fake")
    plan = decoder.ConversionPlan(
        base_time=datetime(2025, 6, 26),
        valid_time=datetime(2025, 7, 1),
        source_npy=source,
        target_dir=tmp_path / "120",
        target_nc=tmp_path / "120" / "output_surface_2025-07-01-00-00.nc",
    )
    calls = []

    def fake_surface(surface_file, file_name, outputs_dir):
        calls.append((Path(surface_file), file_name, Path(outputs_dir)))
        Path(outputs_dir, file_name).write_text("decoded", encoding="utf-8")

    result = decoder.decode_surface_cache_to_nc(plan, surface_decoder=fake_surface)

    assert result.status == "created"
    assert plan.target_nc.read_text(encoding="utf-8") == "decoded"
    assert calls == [(source, "output_surface_2025-07-01-00-00.nc", plan.target_dir)]


def test_decode_surface_cache_to_nc_skips_existing_file(tmp_path):
    source = tmp_path / "timeline_cache" / "output_surface_2025-07-01-00-00.npy"
    source.parent.mkdir()
    source.write_bytes(b"fake")
    target_dir = tmp_path / "120"
    target_dir.mkdir()
    target_nc = target_dir / "output_surface_2025-07-01-00-00.nc"
    target_nc.write_text("existing", encoding="utf-8")
    plan = decoder.ConversionPlan(
        base_time=datetime(2025, 6, 26),
        valid_time=datetime(2025, 7, 1),
        source_npy=source,
        target_dir=target_dir,
        target_nc=target_nc,
    )

    result = decoder.decode_surface_cache_to_nc(plan, surface_decoder=lambda *_: None)

    assert result.status == "skipped_existing"
    assert target_nc.read_text(encoding="utf-8") == "existing"
