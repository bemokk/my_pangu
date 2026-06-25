from types import SimpleNamespace

import torch

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


def test_train_parser_accepts_continuous_future_wind_mode():
    args = train_module.build_arg_parser().parse_args(["--future-wind-mode", "continuous72"])

    assert args.future_wind_mode == "continuous72"


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
