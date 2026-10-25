"""Download ICOADS observations for all 2025 forecast valid dates."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import requests

BASE = 'https://www.ncei.noaa.gov/data/international-comprehensive-ocean-atmosphere/v3/archive/nrt/daily/'
PATTERN = re.compile(r'^icoads3\.0\.2_daily_(updated|preliminary|total_updated|total_preliminary)_d(\d{8})_c(\d{8})\.(nc|dat\.gz)$')
PRIORITY = {'updated': 0, 'preliminary': 1, 'total_updated': 2, 'total_preliminary': 3}


def days(start, end):
    while start <= end:
        yield start
        start += timedelta(days=1)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def request(session, url, **kwargs):
    for attempt in range(3):
        try:
            response = session.get(url, timeout=(30, 180), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def select_month(html, start, end, url):
    selected = {}
    for name in set(re.findall(r'href="([^"]+)"', html)):
        match = PATTERN.fullmatch(name)
        if not match:
            continue
        kind, stamp, cycle, extension = match.groups()
        # NetCDF for ordinary files; total IMMA only as an explicit fallback.
        if extension != ('dat.gz' if kind.startswith('total_') else 'nc'):
            continue
        day = datetime.strptime(stamp, '%Y%m%d').date()
        if not start <= day <= end:
            continue
        item = {'date': str(day), 'kind': kind, 'cycle_date': cycle,
                'format': extension, 'name': name, 'url': url + name}
        rank = (PRIORITY[kind], -int(cycle))
        old = selected.get(str(day))
        if old is None or rank < (PRIORITY[old['kind']], -int(old['cycle_date'])):
            selected[str(day)] = item
    return selected


def plan(session, start, end):
    selected = {}
    for year, month in sorted({(d.year, d.month) for d in days(start, end)}):
        url = f'{BASE}{year}/{month:02d}/'
        with request(session, url) as response:
            selected.update(select_month(response.text, start, end, url))
        print(f'[INDEX] {year}-{month:02d}', flush=True)
    return selected


def validate(path, item):
    """Read wind/coordinate/date arrays, or all gzip bytes (including CRC)."""
    if item['format'] == 'dat.gz':
        with gzip.open(path, 'rb') as stream:
            records = [line for line in stream if line.strip()]
        if not records:
            raise ValueError('empty IMMA archive')
        expected = item['date'].replace('-', '')
        for line in records:
            if len(line) < 84:
                raise ValueError('truncated IMMA record')
            stamp = ''.join(f'{int(line[a:b]):0{width}d}' for a, b, width in [(0, 4, 4), (4, 6, 2), (6, 8, 2)])
            if stamp != expected:
                raise ValueError(f'IMMA date mismatch: {stamp}')
        return {'observations': len(records), 'note': 'IMMA total archive; requires IMMA parser for comparisons'}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with netCDF4.Dataset(path) as ds:
            required = ('date', 'HR', 'lat', 'lon', 'W', 'D', 'ID', 'time')
            missing = set(required) - set(ds.variables)
            if missing:
                raise ValueError(f'missing variables: {sorted(missing)}')
            arrays = {key: ds.variables[key][:] for key in required}
            n = arrays['W'].size
            if n == 0:
                raise ValueError('empty observation file')
            dates = np.asarray(arrays['date']).view('S8').reshape(-1)
            if not np.all(dates == item['date'].replace('-', '').encode()):
                raise ValueError('NetCDF observation date mismatch')
            valid_speed = np.ma.filled(np.isfinite(arrays['W']), False)
            valid_dir = np.ma.filled(np.isfinite(arrays['D']), False)
            return {'observations': int(n), 'wind_speed_count': int(valid_speed.sum()),
                    'wind_pair_count': int((valid_speed & valid_dir).sum())}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def download(session, item, path, retries):
    if path.is_file():
        try:
            return 'skipped', validate(path, item)
        except Exception:
            # Keep an invalid original until its replacement passes validation.
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    for attempt in range(retries):
        try:
            with request(session, item['url'], stream=True) as response:
                if 'text/html' in response.headers.get('Content-Type', '').lower():
                    raise ValueError('server returned HTML')
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                length = response.headers.get('Content-Length')
                if length and not response.headers.get('Content-Encoding') and temporary.stat().st_size != int(length):
                    raise ValueError('Content-Length mismatch')
            stats = validate(temporary, item)
            temporary.replace(path)
            return 'downloaded', stats
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-date', type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument('--end-date', type=date.fromisoformat, default=date(2026, 1, 5))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--max-retry', type=int, default=3)
    parser.add_argument('--list-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    if args.end_date < args.start_date or args.max_retry < 1:
        parser.error('invalid dates or max-retry')
    root = args.root.resolve()
    manifest_path = root / 'manifest.json'
    expected = [str(d) for d in days(args.start_date, args.end_date)]
    if args.verify_only:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        errors = []
        for stamp in expected:
            try:
                row = manifest['records'][stamp]
                path = root / row['path']
                validate(path, row)
                if digest(path) != row['sha256']:
                    raise ValueError('SHA-256 mismatch')
            except Exception as exc:
                errors.append({'date': stamp, 'error': str(exc)})
        atomic_json(root / 'verification.json', {'expected': len(expected), 'valid': len(expected)-len(errors), 'errors': errors})
        print(f'Verified {len(expected)-len(errors)}/{len(expected)}; errors={errors}', flush=True)
        return int(bool(errors))
    with requests.Session() as session:
        session.headers['User-Agent'] = 'pangu-2025-ICOADS-downloader/1.0'
        selected = plan(session, args.start_date, args.end_date)
        missing = [d for d in expected if d not in selected]
        print(f'Plan: expected={len(expected)}, available={len(selected)}, missing={missing}', flush=True)
        if args.list_only:
            for item in sorted(selected.values(), key=lambda r: r['date']):
                print(item['date'], item['kind'], item['name'])
            return int(bool(missing))
        manifest = {'source': BASE, 'start_date': str(args.start_date), 'end_date': str(args.end_date),
                    'records': {}, 'missing_dates': missing, 'failures': []}
        # Re-running safely rediscovers the remote selection and validates existing files.
        for index, stamp in enumerate(expected, 1):
            if stamp not in selected:
                continue
            item = selected[stamp]
            prefix = 'raw' if item['format'] == 'dat.gz' else 'nc'
            path = root / prefix / stamp[:4] / stamp[5:7] / item['name']
            started = time.perf_counter()
            try:
                status, stats = download(session, item, path, args.max_retry)
                manifest['records'][stamp] = dict(item, path=str(path.relative_to(root)),
                    status=status, bytes=path.stat().st_size, sha256=digest(path), **stats)
                print(f'[{index:03d}/{len(expected)}] {stamp} {status} {item["kind"]} {path.stat().st_size/1024**2:.1f} MiB {time.perf_counter()-started:.1f}s', flush=True)
            except Exception as exc:
                manifest['failures'].append({'date': stamp, 'error': str(exc)})
                print(f'[{index:03d}/{len(expected)}] FAILED {stamp}: {exc}', flush=True)
            manifest['updated_at'] = datetime.now().astimezone().isoformat()
            atomic_json(manifest_path, manifest)
        manifest['complete'] = not missing and not manifest['failures']
        manifest['total_bytes'] = sum(row['bytes'] for row in manifest['records'].values())
        atomic_json(manifest_path, manifest)
        print(f'Finished: {len(manifest["records"])}/{len(expected)}, failures={len(manifest["failures"])}; manifest={manifest_path}', flush=True)
        return int(not manifest['complete'])


if __name__ == '__main__':
    raise SystemExit(main())
