from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


OPER_NAME = "data_stream-oper_stepType-instant.nc"
WAVE_NAME = "data_stream-wave_stepType-instant.nc"


@dataclass(frozen=True)
class ExtractedPair:
    archive: Path
    extract_dir: Path
    oper_nc: Path
    wave_nc: Path


def discover_archives(raw_dir: Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    archives = sorted(raw_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"No zip archives found in: {raw_dir}")

    return archives


def extract_archives(raw_dir: Path, extracted_root: Path) -> list[ExtractedPair]:
    extracted_root = Path(extracted_root)
    extracted_root.mkdir(parents=True, exist_ok=True)

    pairs = []
    for archive in discover_archives(raw_dir):
        extract_dir = extracted_root / archive.stem
        oper_nc = extract_dir / OPER_NAME
        wave_nc = extract_dir / WAVE_NAME

        if not (oper_nc.exists() and wave_nc.exists()):
            extract_dir.mkdir(parents=True, exist_ok=True)
            with ZipFile(archive) as zip_file:
                names = set(zip_file.namelist())
                missing = [name for name in (OPER_NAME, WAVE_NAME) if name not in names]
                if missing:
                    raise FileNotFoundError(
                        f"Archive {archive} is missing expected files: {missing}"
                    )
                zip_file.extract(OPER_NAME, extract_dir)
                zip_file.extract(WAVE_NAME, extract_dir)

        pairs.append(
            ExtractedPair(
                archive=archive,
                extract_dir=extract_dir,
                oper_nc=oper_nc,
                wave_nc=wave_nc,
            )
        )

    return pairs
