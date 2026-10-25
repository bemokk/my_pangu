import xarray as xr
import numpy as np
import pandas as pd
import os
from pathlib import Path

# -------------------------- 配置 --------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
nc_gdas = PROJECT_ROOT / "gdas" / "nc" / "processed" / "2018102300" / "upper.nc"
nc_ecmwf = PROJECT_ROOT / "model_input" / "era5" / "2018-10-23-00-00" / "upper.nc"

variables = ["z", "q", "t", "u", "v"]                 # 要比较的变量
time = nc_ecmwf.parent.name
out_csv = time+"upper_compare_by_level.csv"

coord_round = 5   # 坐标取交集时的四舍五入小数位
min_valid_points = 50  # 少于多少点则跳过该层的统计（防止单点/空值问题）
# -----------------------------------------------------------

def normalize_longitude(ds):
    # 把 -180..180 转为 0..360（如果需要）
    lon = ds.longitude.values
    if lon.min() < 0:
        new_lon = np.mod(lon, 360.0)
        ds = ds.assign_coords(longitude=("longitude", new_lon))
        ds = ds.sortby("longitude")
    return ds

def prepare_dataset(path, chunks=None):
    # 若文件很大，传入 chunks={'latitude':100,'longitude':200} 可启用 dask
    ds = xr.open_dataset(path, chunks=chunks) if chunks else xr.open_dataset(path)
    # 保证 latitude 从北到南（降序）
    try:
        if ds.latitude.values[0] < ds.latitude.values[-1]:
            ds = ds.sortby("latitude", ascending=False)
    except Exception:
        pass
    ds = normalize_longitude(ds)
    return ds

def compute_metrics(a, b):
    """
    a, b: numpy arrays (masked with np.nan for invalid points)
    返回 dict: rmse, bias, corr, similarity_percent, n_points
    """
    # flatten并掩去nan
    a1 = a.ravel()
    b1 = b.ravel()
    valid = ~np.isnan(a1) & ~np.isnan(b1)
    n = int(valid.sum())
    if n == 0:
        return None
    a_v = a1[valid].astype(np.float64)
    b_v = b1[valid].astype(np.float64)

    diff = a_v - b_v
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))

    # 相关系数（如果点数小于2则不能算）
    corr = float(np.corrcoef(a_v, b_v)[0,1]) if a_v.size >= 2 else np.nan

    data_min = min(np.nanmin(a_v), np.nanmin(b_v))
    data_max = max(np.nanmax(a_v), np.nanmax(b_v))
    data_range = data_max - data_min
    if data_range == 0:
        similarity = 100.0
    else:
        similarity = float(max(0.0, (1 - rmse / data_range) * 100.0))

    return {"rmse": rmse, "bias": bias, "corr": corr, "similarity_%": similarity, "n_points": n}

def compare_upper(nc1, nc2, variables, out_csv):
    # 如果数据非常大，可以把 chunks={'latitude':100,'longitude':200} 传给 prepare_dataset()
    ds1 = prepare_dataset(nc1, chunks=None)
    ds2 = prepare_dataset(nc2, chunks=None)

    # 获取公共 pressure_level（四舍五入后）
    p1 = np.round(ds1.pressure_level.values.astype(float), 6)
    p2 = np.round(ds2.pressure_level.values.astype(float), 6)
    plevel_common = np.intersect1d(p1, p2)

    if plevel_common.size == 0:
        raise RuntimeError("没有公共 pressure_level，请检查两个文件的 pressure_level 值。")

    # 公共经纬
    lat_common = np.intersect1d(np.round(ds1.latitude.values, coord_round),
                                np.round(ds2.latitude.values, coord_round))
    lon_common = np.intersect1d(np.round(ds1.longitude.values, coord_round),
                                np.round(ds2.longitude.values, coord_round))

    if lat_common.size == 0 or lon_common.size == 0:
        raise RuntimeError("没有公共经纬，请检查坐标或增大 coord_round 值。")

    records = []

    for var in variables:
        if var not in ds1 or var not in ds2:
            print(f"[WARN] 变量 {var} 未在两个文件中同时出现，跳过。")
            continue

        da1 = ds1[var]
        da2 = ds2[var]

        # 确保 pressure_level 轴的值使用相同排序（从高到低或从低到高都行，但交集匹配）
        for plev in plevel_common:
            # 选取该层并公共经纬
            try:
                sel1 = da1.sel(pressure_level=float(plev), latitude=lat_common, longitude=lon_common)
                sel2 = da2.sel(pressure_level=float(plev), latitude=lat_common, longitude=lon_common)
            except Exception as e:
                # 某些 xarray 版本在用 numpy array 选择时可能抛错；尝试按 list
                sel1 = da1.sel(pressure_level=float(plev), latitude=list(lat_common), longitude=list(lon_common))
                sel2 = da2.sel(pressure_level=float(plev), latitude=list(lat_common), longitude=list(lon_common))

            a = sel1.values.astype(np.float64)
            b = sel2.values.astype(np.float64)

            # 将明显的 _FillValue（极大或极小）以及 nan 一并视为无效
            # 这里假设文件里空值已经是 NaN 或者非常大数（>1e30）
            a[np.isinf(a)] = np.nan
            b[np.isinf(b)] = np.nan
            a[np.abs(a) > 1e30] = np.nan
            b[np.abs(b) > 1e30] = np.nan

            metrics = compute_metrics(a, b)
            if metrics is None or metrics["n_points"] < min_valid_points:
                # 少于阈值，记录并跳过
                records.append({
                    "variable": var, "pressure_level": float(plev),
                    "rmse": np.nan, "bias": np.nan, "corr": np.nan,
                    "similarity_%": np.nan, "n_points": int(0)
                })
                continue

            rec = {
                "variable": var,
                "pressure_level": float(plev),
                "rmse": metrics["rmse"],
                "bias": metrics["bias"],
                "corr": metrics["corr"],
                "similarity_%": metrics["similarity_%"],
                "n_points": int(metrics["n_points"])
            }
            records.append(rec)
            print(f"{var} @ {plev} hPa -> sim: {metrics['similarity_%']:.2f}%, rmse: {metrics['rmse']:.4g}, corr: {metrics['corr']:.4f}, n: {metrics['n_points']}")

    df = pd.DataFrame.from_records(records)
    # 保存 CSV
    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df.to_csv(out_csv, index=False)
    print(f"\n结果已保存到: {out_csv}")

    # 同时返回 DataFrame 方便后续处理
    return df

if __name__ == "__main__":
    df = compare_upper(nc_gdas, nc_ecmwf, variables, out_csv)

    # 汇总：对每个变量按层取平均相似度（忽略空值）
    summary = df.groupby("variable")["similarity_%"].mean().reset_index().rename(columns={"similarity_%":"mean_similarity_%"})
    print("\n按变量的平均相似度（各层平均）:")
    print(summary.to_string(index=False))
