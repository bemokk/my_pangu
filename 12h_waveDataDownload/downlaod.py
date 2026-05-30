import os
import requests
import time
import zipfile
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# ================= 核心配置区 =================
AREA_BBOX = [41, 105, 0, 135]  # 中国海域 [北纬, 西经, 南纬, 东经]
UNIFIED_VARS = ['u10', 'v10', 'swh', 'mwp', 'mwd']  # 统一的标准变量名 u10 v10 波高 周期 波向

# === 目录结构配置 ===
BASE_DIR = Path(__file__).resolve().parent

ERA5_RAW_DIR = BASE_DIR / "raw_data" / "era5"
GDAS_RAW_DIR = BASE_DIR / "raw_data" / "gdas"
UNIFIED_DIR = BASE_DIR / "unified_wave_data"


# ==========================================
def get_unified_wave_data(target_utc_time: str):
    target_dt = pd.to_datetime(target_utc_time)
    time_str = target_dt.strftime('%Y%m%d%H')

    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    final_nc_path = UNIFIED_DIR / f"unified_wind_wave_{time_str}_12h.nc"

    # 如果本地已经有清洗好的最终文件，直接读取返回
    if final_nc_path.exists():
        print(f"✅ 发现已存在的本地统一格式数据: {final_nc_path.name}，直接加载。")
        return xr.open_dataset(final_nc_path)

    print(f"\n🚀 开始获取 {target_utc_time} 之前的 12 小时连续数据...")

    # 尝试获取 ERA5
    ds = try_fetch_era5(target_dt)

    # 没有ERA5数据，下载NOAA (GDAS/GFS)
    if ds is None:
        print("⚠️ ERA5 数据获取失败或暂未发布。")
        print("🔄 触发自动降级机制：切换至 NOAA (GDAS/GFS) 近期数据源...")
        ds = fetch_noaa_fallback(target_dt)

    if ds is None:
        raise RuntimeError("❌ 致命错误：所有数据源均无法获取该时间段的数据。")

    # 统一保存为标准的 NetCDF 文件
    print(f"💾 正在将标准化数据持久化保存至: {final_nc_path}")
    ds.to_netcdf(final_nc_path)
    print("✅️ 任务完成！返回标准 Dataset。")

    return ds


def try_fetch_era5(target_dt):
    print("-> [Source 1] 正在请求 Copernicus ERA5 数据库...")
    time_str = target_dt.strftime('%Y%m%d%H')
    base_name = f"era5_raw_{time_str}"

    ERA5_RAW_DIR.mkdir(parents=True, exist_ok=True)
    extract_dir = ERA5_RAW_DIR / base_name
    zip_path = ERA5_RAW_DIR / f"{base_name}.zip"

    times_pd = pd.date_range(end=target_dt - pd.Timedelta(hours=1), periods=12, freq='h')

    try:
        # 1. 检查是否已经解压过，如果没有才去下载/解压
        nc_files = list(extract_dir.glob('*.nc'))

        if not nc_files:
            if not zip_path.exists():
                import cdsapi
                c = cdsapi.Client()
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'netcdf',
                        'variable': [
                            '10m_u_component_of_wind', '10m_v_component_of_wind',
                            'significant_height_of_combined_wind_waves_and_swell',
                            'mean_wave_period', 'mean_wave_direction',
                        ],
                        'year': list(set(times_pd.strftime('%Y'))),
                        'month': list(set(times_pd.strftime('%m'))),
                        'day': list(set(times_pd.strftime('%d'))),
                        'time': list(set(times_pd.strftime('%H:00'))),
                        'area': AREA_BBOX,
                        'grid': ['0.25', '0.25'],
                    },
                    str(zip_path))

            # 解压
            extract_dir.mkdir(parents=True, exist_ok=True)
            print(f"   📦 正在解压至: {extract_dir}")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)

            # 删除 zip
            if zip_path.exists():
                zip_path.unlink()
                print(f"   🗑️ 已清理原始压缩包: {zip_path.name}")

            nc_files = list(extract_dir.glob('*.nc'))

        # 2. 合并数据
        ds = xr.merge([xr.open_dataset(f) for f in nc_files])

        # 3. 维度清洗
        if 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})
        dims_to_drop = [d for d in ['expver', 'number'] if d in ds.dims]
        if dims_to_drop:
            ds = ds.isel({d: 0 for d in dims_to_drop})

        return ds.squeeze().sortby('time')

    except Exception as e:
        print(f"❌  [ERA5 拒绝请求或处理失败] 原因: {e}")
        return None


def fetch_noaa_fallback(target_dt):
    print("-> [Source 2] 正在连接 NOAA NOMADS 服务器...")
    time_str = target_dt.strftime('%Y%m%d%H')

    # 新建对应时间戳的 GDAS/GFS raw 文件夹
    out_path = GDAS_RAW_DIR / f"gdas_raw_{time_str}"
    out_path.mkdir(parents=True, exist_ok=True)

    start_dt = target_dt - pd.Timedelta(hours=12)
    cycle_hour = (start_dt.hour // 6) * 6
    cycle_dt = start_dt.replace(hour=cycle_hour, minute=0, second=0)

    date_str = cycle_dt.strftime('%Y%m%d')
    cycle_str = cycle_dt.strftime('%H')
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{date_str}/{cycle_str}/wave/gridded/"

    times_to_download = pd.date_range(end=target_dt - pd.Timedelta(hours=1), periods=12, freq='h')
    forecast_hours = [int((t - cycle_dt).total_seconds() / 3600) for t in times_to_download]

    print(
        f"   🎯 起报点: {cycle_dt.strftime('%Y-%m-%d %H:%M')} | 时效: f{forecast_hours[0]:03d} -> f{forecast_hours[-1]:03d}")

    if any(h > 120 for h in forecast_hours):
        print("   ❌ NOAA 数据限制: 所需时效超出 120h，存在 3h 步长跳跃。")
        return None

    target_vars = ["UGRD", "VGRD", "HTSGW", "PERPW", "DIRPW"]
    session = requests.Session()
    downloaded_files = []

    for f_hour in forecast_hours:
        file_prefix = f"gfswave.t{cycle_str}z.global.0p25.f{f_hour:03d}.grib2"
        idx_url = f"{base_url}{file_prefix}.idx"
        grib_url = f"{base_url}{file_prefix}"
        grib_file = out_path / f"noaa_raw_{date_str}{cycle_str}_f{f_hour:03d}.grib2"

        if not grib_file.exists() or grib_file.stat().st_size == 0:
            print(f"   ⬇️ 下载 [f{f_hour:03d}]...", end=" ", flush=True)
            try:
                idx_resp = session.get(idx_url, timeout=10)
                idx_resp.raise_for_status()

                lines = idx_resp.text.strip().split('\n')
                byte_ranges = []
                for i in range(len(lines)):
                    parts = lines[i].split(':')
                    if len(parts) >= 4 and parts[3] in target_vars:
                        s_byte = int(parts[1])
                        e_byte = int(lines[i + 1].split(':')[1]) - 1 if i + 1 < len(lines) else ""
                        byte_ranges.append((parts[3], s_byte, e_byte))

                with open(grib_file, "ab") as f:
                    for var_name, s, e in byte_ranges:
                        resp = session.get(grib_url, headers={"Range": f"bytes={s}-{e}"}, stream=True, timeout=15)
                        resp.raise_for_status()
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk: f.write(chunk)
                        time.sleep(0.5)
                print("完成")
            except Exception as e:
                print(f"失败 ({e})")
                return None
            time.sleep(1)

        downloaded_files.append(grib_file)

    print("-> 正在将 NOAA GRIB2 格式翻译为 ERA5 标准格式...")
    ds_list = []

    # === 变量映射字典 ===
    var_mapping = {
        'u': 'u10',
        'v': 'v10',
        'swh': 'swh',
        'htsgw': 'swh',
        'perpw': 'mwp',
        'dirpw': 'mwd'
    }

    for file in downloaded_files:
        ds_temp = xr.open_dataset(file, engine='cfgrib')
        ds_temp = ds_temp.rename({k: v for k, v in var_mapping.items() if k in ds_temp.data_vars})
        ds_temp = ds_temp.sel(
            latitude=slice(AREA_BBOX[0], AREA_BBOX[2]),
            longitude=slice(AREA_BBOX[1], AREA_BBOX[3])
        )
        ds_list.append(ds_temp)

    ds_noaa = xr.concat(ds_list, dim='valid_time')

    coords_to_drop = [c for c in ['step', 'surface', 'time'] if c in ds_noaa.coords]
    if coords_to_drop:
        ds_noaa = ds_noaa.drop_vars(coords_to_drop)

    if 'valid_time' in ds_noaa.coords:
        ds_noaa = ds_noaa.rename({'valid_time': 'time'})
    elif 'valid_time' in ds_noaa.dims:
        ds_noaa = ds_noaa.rename_dims({'valid_time': 'time'})

    return ds_noaa[UNIFIED_VARS].sortby('time')


# ================= 新增：可视化绘图模块 =================
def plot_unified_dataset(ds, target_utc_time):
    #接收统一清洗后的 Dataset，绘制 5x12 的特征可视化矩阵图

    target_dt = pd.to_datetime(target_utc_time)
    time_str = target_dt.strftime('%Y%m%d%H')

    # 将图片保存在统一数据目录下
    img_path = UNIFIED_DIR / f"visualization_{time_str}.jpg"

    print(f"\n🎨 正在生成 5x12 可视化矩阵...")
    var_config = [
        ('u10', 'U', 'RdBu_r'),
        ('v10', 'V', 'RdBu_r'),
        ('swh', 'HS', 'viridis'),
        ('mwp', 'PERIOD', 'plasma'),
        ('mwd', 'DIRECTION', 'twilight')
    ]

    fig, axes = plt.subplots(nrows=5, ncols=12, figsize=(38, 16))

    for i, (var, title, cmap) in enumerate(var_config):
        for j in range(12):
            ax = axes[i, j]
            if var in ds.data_vars:
                # 提取特定时刻的切片
                data_slice = ds[var].isel(time=j)
                # 提取时间戳用于标题
                current_time_str = pd.to_datetime(data_slice.time.values).strftime('%Y-%m-%dT%H:00')

                im = data_slice.plot(ax=ax, cmap=cmap, add_colorbar=False)

                ax.set_title(f"{title} | {current_time_str}", fontsize=10)
                ax.set_xlabel("X", fontsize=8)
                ax.set_ylabel("Y", fontsize=8)

                # 为每个子图添加独立的 Colorbar
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
            else:
                ax.set_title(f"Missing: {var}")

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 图像已保存至: {img_path}")


# ================= 运行测试区 =================
if __name__ == "__main__":
    #目标时间
    target = '2026-03-29 12:00'

    # 1. 下载并处理数据
    final_dataset = get_unified_wave_data(target_utc_time=target)

    # 2. 绘制 5x12 可视化图
    plot_unified_dataset(final_dataset, target)
