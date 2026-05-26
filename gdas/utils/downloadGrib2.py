import requests
from tqdm import tqdm
import os


def download_with_progress(url, save_path):
    """
    使用 tqdm 显示下载进度条
    """
    filename = url.split("/")[-1]
    if save_path is None:
        # 从 URL 中提取文件名
        save_path = filename
    else:
        save_path += filename

    # 发送 GET 请求，stream=True 支持流式下载
    response = requests.get(url, stream=True)

    # 获取文件总大小（字节）
    total_size = int(response.headers.get('content-length', 0))

    # 初始化进度条
    with open(save_path, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=8192):
            size = file.write(data)
            progress_bar.update(size)

    print(f"\n下载完成！文件保存为: {save_path}")
    print(f"文件大小: {total_size / (1024 * 1024):.2f} MB")
    return save_path


# 使用示例
url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gdas.20251215/00/atmos/gdas.t00z.pgrb2.0p25.f000.idx"
time = (url.split("/")[-4]).split(".")[-1]
download_with_progress(url,"grib2/"+time)