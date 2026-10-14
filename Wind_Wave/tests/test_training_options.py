from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wind_wave.evaluate import _load_model_state
import wind_wave.train as train_module


def test_train_parser_accepts_regularization_and_early_stopping_options():
    args = train_module.build_arg_parser().parse_args(
        [
            "--dropout",
            "0.1",
            "--weight-decay",
            "1e-4",
            "--early-stopping-patience",
            "3",
        ]
    )

    assert args.dropout == 0.1
    assert args.weight_decay == 1e-4
    assert args.early_stopping_patience == 3


def test_train_parser_accepts_cuda_throughput_options():
    args = train_module.build_arg_parser().parse_args(
        [
            "--precision",
            "bf16",
            "--compile-model",
            "--pin-memory",
            "--persistent-workers",
            "--prefetch-factor",
            "4",
            "--log-every",
            "10",
            "--fast-in-memory-dataset",
        ]
    )

    assert args.precision == "bf16"
    assert args.compile_model is True
    assert args.pin_memory is True
    assert args.persistent_workers is True
    assert args.prefetch_factor == 4
    assert args.log_every == 10
    assert args.fast_in_memory_dataset is True


def test_train_parser_accepts_epoch_pause_seconds():
    args = train_module.build_arg_parser().parse_args(["--epoch-pause-seconds", "30"])

    assert args.epoch_pause_seconds == 30.0


def test_train_parser_accepts_lossless_resume_options(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    args = train_module.build_arg_parser().parse_args(
        [
            "--seed",
            "2026",
            "--resume-checkpoint",
            str(checkpoint),
            "--checkpoint-every-steps",
            "250",
            "--step-pause-every",
            "500",
            "--step-pause-seconds",
            "15",
        ]
    )

    assert args.seed == 2026
    assert args.resume_checkpoint == checkpoint
    assert args.checkpoint_every_steps == 250
    assert args.step_pause_every == 500
    assert args.step_pause_seconds == 15.0


def test_train_parser_accepts_continuous_future_wind_mode():
    args = train_module.build_arg_parser().parse_args(["--future-wind-mode", "continuous72"])

    assert args.future_wind_mode == "continuous72"


def test_train_parser_accepts_purged_circular_direction_experiment_options():
    args = train_module.build_arg_parser().parse_args(
        [
            "--model-variant",
            "m2-wave0-direct-unitdir",
            "--split-embargo-hours",
            "96",
            "--loss-mode",
            "scalar-circular",
            "--direction-loss-weight",
            "0.5",
        ]
    )

    assert args.model_variant == "m2-wave0-direct-unitdir"
    assert args.split_embargo_hours == 96
    assert args.loss_mode == "scalar-circular"
    assert args.direction_loss_weight == 0.5


class ModelWithTargetStats(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.register_buffer("target_mean", torch.zeros(4))
        self.register_buffer("target_std", torch.ones(4))


def test_load_model_state_accepts_legacy_checkpoint_without_target_stats():
    model = ModelWithTargetStats()
    legacy_state = {
        key: value
        for key, value in model.state_dict().items()
        if key not in {"target_mean", "target_std"}
    }

    _load_model_state(model, legacy_state)

    assert torch.equal(model.target_mean, torch.zeros(4))
    assert torch.equal(model.target_std, torch.ones(4))


def test_load_model_state_rejects_other_missing_keys():
    model = ModelWithTargetStats()
    incomplete_state = {
        key: value
        for key, value in model.state_dict().items()
        if key not in {"linear.weight", "target_mean", "target_std"}
    }

    with pytest.raises(RuntimeError, match="linear.weight"):
        _load_model_state(model, incomplete_state)


def test_pause_after_epoch_sleeps_between_epochs_only():
    sleeps = []
    messages = []

    train_module._pause_after_epoch(
        epoch=1,
        total_epochs=3,
        seconds=30,
        sleeper=sleeps.append,
        logger=messages.append,
    )

    assert sleeps == [30]
    assert messages == ["epoch=1 pause_seconds=30.0"]

    train_module._pause_after_epoch(
        epoch=3,
        total_epochs=3,
        seconds=30,
        sleeper=sleeps.append,
        logger=messages.append,
    )

    assert sleeps == [30]


def test_pause_after_step_sleeps_only_at_internal_intervals():
    sleeps = []
    messages = []

    for step in (499, 500, 1000, 1200):
        train_module._pause_after_step(
            step=step,
            total_steps=1200,
            every_steps=500,
            seconds=15,
            sleeper=sleeps.append,
            logger=messages.append,
        )

    assert sleeps == [15, 15]
    assert messages == [
        "step=500/1200 pause_seconds=15.0",
        "step=1000/1200 pause_seconds=15.0",
    ]


def test_build_model_adds_dropout_to_v2_heads():
    args = SimpleNamespace(
        model_variant="m2-wave0-direct",
        hidden_channels=4,
        dropout=0.1,
    )

    model = train_module._build_model(args, lead_count=5)

    dropouts = [module for module in model.modules() if isinstance(module, torch.nn.Dropout2d)]
    assert dropouts
    assert {module.p for module in dropouts} == {0.1}


def test_build_model_configures_continuous_future_wind_mode():
    args = SimpleNamespace(
        model_variant="m2-direct",
        hidden_channels=4,
        dropout=0.0,
        future_wind_mode="continuous72",
    )

    model = train_module._build_model(args, lead_count=5)

    assert model.future_wind_mode == "continuous72"


def test_build_model_configures_direct_unit_direction_variant():
    args = SimpleNamespace(
        model_variant="m2-wave0-direct-unitdir",
        hidden_channels=4,
        dropout=0.0,
        future_wind_mode="target",
    )

    model = train_module._build_model(args, lead_count=5)

    assert model.use_wave0 is True
    assert model.residual is False
    assert model.normalize_direction_output is True


def test_build_optimizer_uses_configured_weight_decay():
    model = torch.nn.Linear(2, 1)
    args = SimpleNamespace(learning_rate=1e-3, weight_decay=1e-4)

    optimizer = train_module._build_optimizer(model, args)

    assert optimizer.param_groups[0]["weight_decay"] == 1e-4


def test_early_stopping_triggers_after_patience_epochs_without_improvement():
    assert train_module._should_stop_early(epochs_without_improvement=2, patience=3) is False
    assert train_module._should_stop_early(epochs_without_improvement=3, patience=3) is True
    assert train_module._should_stop_early(epochs_without_improvement=99, patience=0) is False


def test_autocast_dtype_uses_tensor_core_friendly_precision_on_cuda():
    assert train_module._autocast_dtype("bf16", torch.device("cuda")) is torch.bfloat16
    assert train_module._autocast_dtype("fp16", torch.device("cuda")) is torch.float16
    assert train_module._autocast_dtype("tf32", torch.device("cuda")) is None
    assert train_module._autocast_dtype("bf16", torch.device("cpu")) is None


def test_build_loader_kwargs_enables_pinned_prefetching_only_when_supported():
    args = SimpleNamespace(
        batch_size=16,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    kwargs = train_module._build_loader_kwargs(args, shuffle=True)

    assert kwargs["batch_size"] == 16
    assert kwargs["shuffle"] is True
    assert kwargs["num_workers"] == 2
    assert kwargs["pin_memory"] is True
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 4


def test_build_loader_kwargs_skips_worker_only_options_for_single_process_loading():
    args = SimpleNamespace(
        batch_size=16,
        num_workers=0,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    kwargs = train_module._build_loader_kwargs(args, shuffle=False)

    assert kwargs["pin_memory"] is True
    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs


def test_epoch_data_generator_repeats_the_same_order_for_resume():
    first = torch.randperm(32, generator=train_module._epoch_data_generator(1337, 4))
    resumed = torch.randperm(32, generator=train_module._epoch_data_generator(1337, 4))
    next_epoch = torch.randperm(32, generator=train_module._epoch_data_generator(1337, 5))

    assert torch.equal(first, resumed)
    assert not torch.equal(first, next_epoch)


def test_full_checkpoint_round_trip_preserves_optimizer_and_progress(tmp_path):
    args = train_module.build_arg_parser().parse_args(
        ["--batch-size", "4", "--checkpoint-every-steps", "2"]
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss = model(torch.ones(2, 2)).sum()
    loss.backward()
    optimizer.step()
    scaler = train_module._build_grad_scaler("fp32", torch.device("cpu"))
    stats = train_module.NormalizationStats(
        input_mean=np.zeros(2, dtype=np.float32),
        input_std=np.ones(2, dtype=np.float32),
        target_mean=np.zeros(4, dtype=np.float32),
        target_std=np.ones(4, dtype=np.float32),
        input_names=("u10", "v10"),
        target_names=("swh", "mwp", "cos_mwd", "sin_mwd"),
    )
    checkpoint = train_module._build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        stats=stats,
        lead_hours=(6, 12, 24, 48, 72),
        args=args,
        epoch=4,
        step_in_epoch=250,
        epoch_complete=False,
        epoch_loss_sum=12.5,
        epoch_loss_count=250,
        train_samples=1000,
        train_seconds=42.0,
        train_log=[{"epoch": 3, "train_loss": 0.5}],
        metrics_rows=[{"epoch": 3, "split": "validation"}],
        best_val=0.4,
        epochs_without_improvement=1,
    )
    path = tmp_path / "latest.pt"

    train_module._atomic_torch_save(checkpoint, path)
    loaded = train_module._load_checkpoint(str(path))

    assert loaded["checkpoint_version"] == train_module.CHECKPOINT_VERSION
    assert loaded["optimizer_state_dict"]["state"]
    assert loaded["progress"]["epoch"] == 4
    assert loaded["progress"]["step_in_epoch"] == 250
    assert loaded["progress"]["epoch_complete"] is False
    assert loaded["train_log"] == [{"epoch": 3, "train_loss": 0.5}]
    assert not path.with_name("latest.pt.tmp").exists()
    train_module._validate_resume_checkpoint(loaded, args, (6, 12, 24, 48, 72))


def test_resume_rejects_a_batch_size_change(tmp_path):
    args = train_module.build_arg_parser().parse_args(["--batch-size", "4"])
    changed_args = train_module.build_arg_parser().parse_args(["--batch-size", "8"])
    checkpoint = {
        "checkpoint_version": train_module.CHECKPOINT_VERSION,
        "optimizer_state_dict": {},
        "scaler_state_dict": {},
        "progress": {},
        "rng_state": {},
        "training_config": train_module._checkpoint_training_config(
            args, (6, 12, 24, 48, 72)
        ),
    }

    with pytest.raises(ValueError, match="batch_size"):
        train_module._validate_resume_checkpoint(
            checkpoint, changed_args, (6, 12, 24, 48, 72)
        )


def test_checkpoint_resume_matches_uninterrupted_adam_step(tmp_path):
    train_module._seed_everything(77)
    args = train_module.build_arg_parser().parse_args(["--seed", "77"])
    stats = train_module.NormalizationStats(
        input_mean=np.zeros(2, dtype=np.float32),
        input_std=np.ones(2, dtype=np.float32),
        target_mean=np.zeros(4, dtype=np.float32),
        target_std=np.ones(4, dtype=np.float32),
        input_names=("u10", "v10"),
        target_names=("swh", "mwp", "cos_mwd", "sin_mwd"),
    )

    def build_model():
        return torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(4, 1),
        )

    def train_step(model, optimizer, features, targets):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(features), targets)
        loss.backward()
        optimizer.step()

    features1 = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.25, -1.0]])
    targets1 = torch.tensor([[0.2], [-0.1]])
    features2 = torch.tensor([[0.1, -2.0, 1.5], [1.25, 0.0, -0.5]])
    targets2 = torch.tensor([[0.7], [0.3]])

    uninterrupted = build_model()
    uninterrupted_optimizer = torch.optim.Adam(uninterrupted.parameters(), lr=1e-3)
    scaler = train_module._build_grad_scaler("fp32", torch.device("cpu"))
    train_step(uninterrupted, uninterrupted_optimizer, features1, targets1)
    checkpoint = train_module._build_training_checkpoint(
        model=uninterrupted,
        optimizer=uninterrupted_optimizer,
        scaler=scaler,
        stats=stats,
        lead_hours=(6, 12, 24, 48, 72),
        args=args,
        epoch=1,
        step_in_epoch=1,
        epoch_complete=False,
        epoch_loss_sum=1.0,
        epoch_loss_count=1,
        train_samples=2,
        train_seconds=1.0,
        train_log=[],
        metrics_rows=[],
        best_val=float("inf"),
        epochs_without_improvement=0,
    )
    path = tmp_path / "resume.pt"
    train_module._atomic_torch_save(checkpoint, path)

    train_step(uninterrupted, uninterrupted_optimizer, features2, targets2)
    expected = [parameter.detach().clone() for parameter in uninterrupted.parameters()]

    loaded = train_module._load_checkpoint(path)
    resumed = build_model()
    resumed.load_state_dict(loaded["model_state_dict"])
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    train_module._restore_rng_state(loaded["rng_state"])
    train_step(resumed, resumed_optimizer, features2, targets2)

    for expected_parameter, resumed_parameter in zip(expected, resumed.parameters()):
        assert torch.equal(expected_parameter, resumed_parameter)


def test_write_preview_casts_bfloat16_predictions_to_numpy_supported_dtype(tmp_path):
    predictions = torch.ones((1, 1, 1, 1, 1), dtype=torch.bfloat16)
    targets = torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32)
    batch = {"t0": ["2016-01-01T00:00:00"]}

    train_module._write_preview(
        tmp_path / "preview.npz",
        tmp_path / "metadata.csv",
        predictions,
        targets,
        batch,
    )

    loaded = train_module.np.load(tmp_path / "preview.npz")
    assert loaded["predictions"].dtype == train_module.np.float32


def test_maybe_compile_model_skips_when_triton_is_unavailable(monkeypatch):
    model = torch.nn.Linear(2, 1)
    args = SimpleNamespace(compile_model=True)
    monkeypatch.setattr(train_module, "_triton_available", lambda: False)

    compiled = train_module._maybe_compile_model(model, args, torch.device("cuda"))

    assert compiled is model
