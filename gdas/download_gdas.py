from pathlib import Path
from datetime import date, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# 保存目录
OUT_DIR = Path(r"E:\GDAS\202507_00UTC")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GDEX HTTPS 下载地址模板
BASE_URL = "https://data.gdex.ucar.edu/d083003/{year}/{yyyymm}/{filename}"

START_DATE = date(2025, 7, 8)
END_DATE = date(2025, 7, 8)


def make_session():
    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"]
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "Mozilla/5.0 GDAS-FNL downloader"
    })

    return session


def daterange(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def get_remote_file_size(session, url):
    """
    获取远程文件大小，用于判断是否已经下载完整
    """
    try:
        r = session.head(url, timeout=(20, 60), allow_redirects=True)
        r.raise_for_status()
        size = r.headers.get("Content-Length")
        return int(size) if size is not None else None
    except Exception:
        return None


def download_file(session, url, out_path):
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    remote_size = get_remote_file_size(session, url)

    # 如果正式文件已经存在，并且大小一致，则跳过
    if out_path.exists() and remote_size is not None:
        local_size = out_path.stat().st_size
        if local_size == remote_size:
            print(f"[跳过] 已完整下载: {out_path.name}")
            return

    # 如果正式文件存在但大小不一致，改名为 .part，继续尝试断点续传
    if out_path.exists() and not tmp_path.exists():
        out_path.rename(tmp_path)

    downloaded = 0
    headers = {}
    mode = "wb"

    # 断点续传
    if tmp_path.exists():
        downloaded = tmp_path.stat().st_size
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"
            mode = "ab"

    print(f"\n[下载] {out_path.name}")

    with session.get(url, stream=True, timeout=(20, 180), headers=headers) as r:
        # 如果服务器不支持断点续传，会返回 200，此时重新下载
        if r.status_code == 200 and downloaded > 0:
            print("[提示] 服务器未响应断点续传，将重新下载该文件")
            downloaded = 0
            mode = "wb"

        r.raise_for_status()

        content_length = r.headers.get("Content-Length")

        if content_length is not None:
            total_size = int(content_length) + downloaded
        else:
            total_size = remote_size

        with open(tmp_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=downloaded,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=out_path.name,
                ncols=100
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    tmp_path.rename(out_path)
    print(f"[完成] {out_path.name}")


def main():
    session = make_session()

    for d in daterange(START_DATE, END_DATE):
        year = d.strftime("%Y")
        yyyymm = d.strftime("%Y%m")
        yyyymmdd = d.strftime("%Y%m%d")

        filename = f"gdas1.fnl0p25.{yyyymmdd}00.f00.grib2"

        url = BASE_URL.format(
            year=year,
            yyyymm=yyyymm,
            filename=filename
        )

        out_path = OUT_DIR / filename

        try:
            download_file(session, url, out_path)
        except Exception as e:
            print(f"[失败] {filename}")
            print(f"       {e}")


if __name__ == "__main__":
    main()