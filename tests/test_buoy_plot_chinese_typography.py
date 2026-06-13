from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_SCRIPTS = [
    PROJECT_ROOT / "Buoy/plots/plot_china_sea_hex_counts.py",
    PROJECT_ROOT / "Buoy/plots/plot_wind_speed_metrics_figure2.py",
    PROJECT_ROOT / "Buoy/plots/plot_wind_speed_beaufort_metrics.py",
    PROJECT_ROOT / "Buoy/plots/plot_wind_speed_beaufort_sample_counts.py",
    PROJECT_ROOT / "Buoy/plots/plot_spatial_hex_best_rmse.py",
    PROJECT_ROOT / "Buoy/plots/plot_wipha_track_forecast_error.py",
]
EXPERIMENT_LABEL_SCRIPTS = {
    "plot_wind_speed_metrics_figure2.py",
    "plot_wind_speed_beaufort_metrics.py",
    "plot_spatial_hex_best_rmse.py",
    "plot_wipha_track_forecast_error.py",
}
CHINESE_EXPERIMENT_LABELS = {"ERA5实时场", "ERA5延迟5天预报", "GDAS实时预报"}


def assignments(tree: ast.AST) -> dict[str, object]:
    values = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def called_attributes(tree: ast.AST) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_target_scripts_expose_editable_chinese_typography_configuration():
    for script in TARGET_SCRIPTS:
        tree = ast.parse(script.read_text(encoding="utf-8-sig"))
        values = assignments(tree)

        assert values.get("FONT_SCALE") == 1.25, script.name
        assert values.get("FONT_FAMILY"), script.name
        assert isinstance(values.get("TEXT_LABELS"), dict), script.name

        calls = called_attributes(tree)
        assert "suptitle" not in calls, script.name
        assert "text" not in {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "fig"
        }, script.name

        if script.name in EXPERIMENT_LABEL_SCRIPTS:
            labels = set(values["TEXT_LABELS"].values())
            assert labels.intersection(CHINESE_EXPERIMENT_LABELS), script.name
