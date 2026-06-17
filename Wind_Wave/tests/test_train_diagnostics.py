import numpy as np
import torch

from wind_wave.dataset import NormalizationStats
from wind_wave.train import _evaluate_persistence_loader, _plot_training_curve


def test_evaluate_persistence_loader_reports_metrics_by_lead():
    stats = NormalizationStats.identity(
        input_names=("u10", "v10"),
        target_names=("swh", "mwp", "cos_mwd", "sin_mwd"),
    )
    persistence = torch.tensor(
        [[[[[1.0]], [[4.0]], [[1.0]], [[0.0]]], [[[1.0]], [[4.0]], [[1.0]], [[0.0]]]]]
    )
    targets = torch.tensor(
        [[[[[2.0]], [[6.0]], [[0.0]], [[1.0]]], [[[3.0]], [[7.0]], [[-1.0]], [[0.0]]]]]
    )
    loader = [{"persistence": persistence, "targets": targets}]

    loss, rows = _evaluate_persistence_loader(
        loader,
        stats,
        torch.device("cpu"),
        lead_hours=(6, 12),
    )

    assert np.isclose(loss, 3.0)
    assert rows[0]["lead_hour"] == 6
    assert np.isclose(rows[0]["rmse_swh"], 1.0)
    assert np.isclose(rows[0]["rmse_mwp"], 2.0)
    assert np.isclose(rows[0]["mae_mwd_degrees"], 90.0)
    assert rows[1]["lead_hour"] == 12
    assert np.isclose(rows[1]["rmse_swh"], 2.0)
    assert np.isclose(rows[1]["rmse_mwp"], 3.0)
    assert np.isclose(rows[1]["mae_mwd_degrees"], 180.0)


def test_evaluate_persistence_loader_keeps_rows_when_direction_is_all_nan():
    stats = NormalizationStats.identity(
        input_names=("u10", "v10"),
        target_names=("swh", "mwp", "cos_mwd", "sin_mwd"),
    )
    persistence = torch.tensor([[[[[1.0]], [[4.0]], [[float("nan")]], [[float("nan")]]]]])
    targets = torch.tensor([[[[[2.0]], [[6.0]], [[float("nan")]], [[float("nan")]]]]])
    loader = [{"persistence": persistence, "targets": targets}]

    loss, rows = _evaluate_persistence_loader(
        loader,
        stats,
        torch.device("cpu"),
        lead_hours=(6,),
    )

    assert np.isfinite(loss)
    assert rows[0]["lead_hour"] == 6
    assert np.isclose(rows[0]["rmse_swh"], 1.0)
    assert np.isclose(rows[0]["rmse_mwp"], 2.0)
    assert np.isnan(rows[0]["mae_mwd_degrees"])


def test_plot_training_curve_writes_nonempty_png(tmp_path):
    path = tmp_path / "training_curve.png"

    _plot_training_curve(
        path,
        [
            {"epoch": 1, "train_loss": 1.2, "val_loss": 1.4},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 1.1},
        ],
    )

    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes().startswith(b"\x89PNG")
