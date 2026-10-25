from __future__ import annotations

import argparse
import calendar
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

from paths import icoads_root


BASE_URL_TEMPLATE = (
    "https://www.ncei.noaa.gov/data/"
    "international-comprehensive-ocean-atmosphere/v3/archive/nrt/daily/{year}/{month:02d}/"
)
HEADERS = {"User-Agent": "Mozilla/5.0"}


def is_valid_netcdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 1000:
        return False
    with path.open("rb") as handle:
        signature = handle.read(8)
    return signature.startswith(b"CDF") or signature.startswith(b"\x89HDF")


def download_nc(url: str, local_path: Path, max_retry: int = 3) -> Path:
    if is_valid_netcdf(local_path):
        print(f"Local NetCDF exists, skip: {local_path.name}")
        return local_path

    if local_path.exists():
        print(f"Existing file is incomplete or invalid, redownload: {local_path.name}")
        local_path.unlink()

    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    for attempt in range(1, max_retry + 1):
        try:
            print(f"Download attempt {attempt}: {url}")
            with requests.get(url, headers=HEADERS, stream=True, timeout=(20, 300)) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"Unexpected HTTP status: {response.status_code}")

                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type:
                    raise RuntimeError(f"Remote returned HTML instead of NetCDF: {content_type}")

                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            print(
                                f"\r  {downloaded / 1024 / 1024:.2f} MB / "
                                f"{total / 1024 / 1024:.2f} MB ({pct:.1f}%)",
                                end="",
                            )
                        else:
                            print(f"\r  {downloaded / 1024 / 1024:.2f} MB", end="")
                print()

            if not is_valid_netcdf(tmp_path):
                raise RuntimeError("Downloaded file is not a valid NetCDF file.")

            tmp_path.replace(local_path)
            print(f"Downloaded: {local_path}")
            return local_path
        except Exception as exc:  # noqa: BLE001 - retry transient network failures.
            print(f"Download failed for {local_path.name}: {exc}")
            if tmp_path.exists():
                tmp_path.unlink()
            time.sleep(3)

    raise RuntimeError(f"Failed after {max_retry} attempts: {url}")


def list_daily_updated_files(year: int, month: int, start_day: int, end_day: int) -> list[str]:
    base_url = BASE_URL_TEMPLATE.format(year=year, month=month)
    html = requests.get(base_url, headers=HEADERS, timeout=60).text

    pattern = rf"icoads3\.0\.2_daily_updated_d{year}{month:02d}(\d{{2}})_c\d{{8}}\.nc"
    names = []
    for match in re.finditer(pattern, html):
        day = int(match.group(1))
        if start_day <= day <= end_day:
            names.append(match.group(0))
    return sorted(set(names))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download ICOADS daily_updated NetCDF files.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--month", type=int, default=8)
    parser.add_argument("--start-day", type=int, default=1)
    parser.add_argument("--end-day", type=int, default=3)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--max-retry", type=int, default=3)
    parser.add_argument("--list-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, days_in_month = calendar.monthrange(args.year, args.month)
    if not (1 <= args.start_day <= args.end_day <= days_in_month):
        raise ValueError(
            f"Invalid day range for {args.year}-{args.month:02d}: "
            f"{args.start_day}..{args.end_day}"
        )

    root = args.out_root if args.out_root else icoads_root(args.year, args.month)
    nc_dir = root / "nc"
    nc_dir.mkdir(parents=True, exist_ok=True)

    base_url = BASE_URL_TEMPLATE.format(year=args.year, month=args.month)
    nc_files = list_daily_updated_files(args.year, args.month, args.start_day, args.end_day)
    print(f"Remote directory: {base_url}")
    print(f"Output directory: {nc_dir}")
    print(f"Found {len(nc_files)} files for days {args.start_day}..{args.end_day}")

    if not nc_files:
        raise RuntimeError("No matching ICOADS daily_updated NetCDF files found.")

    for name in nc_files:
        print(name)

    if args.list_only:
        return

    for name in nc_files:
        download_nc(urljoin(base_url, name), nc_dir / name, max_retry=args.max_retry)

    print("Download complete.")


if __name__ == "__main__":
    main()
