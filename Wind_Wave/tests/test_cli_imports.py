from wind_wave.evaluate import build_arg_parser as build_eval_parser
from wind_wave.train import build_arg_parser as build_train_parser


def test_train_parser_defaults_match_seq2seq_design():
    args = build_train_parser().parse_args([])

    assert args.history_hours == 24
    assert args.lead_hours == "6,12,24,48,72"
    assert args.batch_size == 1
    assert args.input_region == "5,45,95,150"
    assert args.output_region == "15,40,105,135"
    assert args.preload_spatial is False
    assert args.model_variant == "m1"
    assert args.run_name is None


def test_train_parser_accepts_v2_variants_and_run_name():
    args = build_train_parser().parse_args(
        ["--model-variant", "m2-wave0-residual", "--run-name", "m2_wave0_residual"]
    )

    assert args.model_variant == "m2-wave0-residual"
    assert args.run_name == "m2_wave0_residual"


def test_train_parser_accepts_converted_multi_year_data():
    args = build_train_parser().parse_args(
        ["--data-source", "converted", "--years", "2016:2024", "--converted-dir", "data/converted"]
    )

    assert args.data_source == "converted"
    assert args.years == "2016:2024"
    assert str(args.converted_dir) == "data\\converted" or str(args.converted_dir) == "data/converted"


def test_eval_parser_accepts_checkpoint_argument():
    args = build_eval_parser().parse_args(["--checkpoint", "model.pt"])

    assert args.checkpoint == "model.pt"
