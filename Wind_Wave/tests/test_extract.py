from wind_wave.extract import OPER_NAME, WAVE_NAME, extract_archives


def test_extract_archives_reuses_complete_extracted_pairs_without_raw_archives(tmp_path):
    raw_dir = tmp_path / "missing-raw"
    extracted_root = tmp_path / "extracted"
    pair_dir = extracted_root / "archive-b"
    pair_dir.mkdir(parents=True)
    (pair_dir / OPER_NAME).write_bytes(b"wind")
    (pair_dir / WAVE_NAME).write_bytes(b"wave")

    pairs = extract_archives(raw_dir, extracted_root)

    assert len(pairs) == 1
    assert pairs[0].extract_dir == pair_dir
    assert pairs[0].oper_nc == pair_dir / OPER_NAME
    assert pairs[0].wave_nc == pair_dir / WAVE_NAME
