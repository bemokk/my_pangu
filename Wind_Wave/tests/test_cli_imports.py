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


def test_eval_parser_accepts_checkpoint_argument():
    args = build_eval_parser().parse_args(["--checkpoint", "model.pt"])

    assert args.checkpoint == "model.pt"
