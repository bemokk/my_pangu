from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Buoy"))

from plots import plot_wind_speed_metrics_figure2 as speed_plot  # noqa: E402


def test_load_metrics_can_append_lead_zero_rows(monkeypatch, tmp_path):
    csv_path = tmp_path / "wind_speed_metrics_by_lead.csv"
    pd.DataFrame(
        [
            {
                "dataset": "era5_realtime",
                "lead_hour": 3,
                "rmse": 2.0,
                "mae": 1.0,
                "corr": 0.8,
            }
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(
        speed_plot,
        "build_lead_zero_metric_rows",
        lambda variable: pd.DataFrame(
            [
                {
                    "dataset": "era5_realtime",
                    "dataset_label": "ERA5 realtime",
                    "lead_hour": 0,
                    "variable": variable,
                    "n": 10,
                    "rmse": 1.0,
                    "mae": 0.5,
                    "bias": 0.0,
                    "corr": 0.9,
                    "pred_mean": 3.0,
                    "obs_mean": 3.0,
                    "diff_std": 1.0,
                }
            ]
        ),
    )

    result = speed_plot.load_metrics(csv_path, include_lead_zero=True)

    assert result["lead_hour"].tolist() == [0, 3]
    assert result["rmse"].tolist() == [1.0, 2.0]


def test_plot_style_has_chinese_font_fallback_and_darker_lines():
    speed_plot.set_plot_style()

    assert "Times New Roman" in speed_plot.FONT_FAMILY
    assert {"SimSun", "SimHei", "Microsoft YaHei"}.issubset(speed_plot.FONT_FAMILY)

    colors = {dataset: style["color"] for dataset, style in speed_plot.DATASET_STYLES.items()}
    assert colors == {
        "era5_realtime": "#9E2F33",
        "era5_lagged_5d": "#244C8F",
        "gdas_forecast": "#2F7D45",
    }
