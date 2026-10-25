# -*- coding: utf-8 -*-
"""
monthly_error_statistics.py

功能：
读取 oneclick_compare_pangu_era5_china_sea.py 输出的整月 CSV，
统计整月平均误差。

推荐输入：
batch_all_metrics_china_sea.csv

主要输出：
1. monthly_by_lead_variable.csv
   每个变量、每个预报时效的整月平均误差

2. monthly_by_variable_all_leads.csv
   每个变量在所有预报时效上的整月总体误差

3. monthly_surface_by_lead.csv
   近地面变量按时效统计

4. monthly_upper_by_level.csv
   高空变量按气压层、时效统计
"""

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 1. 用户配置区：只需要改这里
# ============================================================

# 整月统计结果 CSV
INPUT_CSV = Path(
    PROJECT_ROOT
    / "src"
    / "comparison_results"
    / "batch_gdas_2025070100_2025073100_china_sea"
    / "batch_all_metrics_china_sea.csv"
)

# 输出目录
OUT_DIR = INPUT_CSV.parent / "monthly_error_statistics"

# 是否只统计这些目标月份的数据
# None 表示不额外筛选
TARGET_MONTH = "2025-07"

# 如果只想统计某一种数据类型，可设置为 "gdas" 或 "era5"
# None 表示不筛选
DATA_TYPE_FILTER = None
# DATA_TYPE_FILTER = "gdas"
# DATA_TYPE_FILTER = "era5"

# 是否输出 Excel
SAVE_EXCEL = True


# ============================================================
# 2. 工具函数
# ============================================================

def safe_weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if valid.sum() == 0:
        return np.nan

    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def pooled_rmse(rmse_values, n_values):
    """
    根据每一天/每个样本块的 RMSE 和样本数 n，合成整月 RMSE。

    原理：
    rmse = sqrt(mean(error^2))

    若每一天已有：
    rmse_i = sqrt(mean(error_i^2))

    则整月合成：
    monthly_rmse = sqrt(sum(n_i * rmse_i^2) / sum(n_i))
    """

    rmse_values = np.asarray(rmse_values, dtype=float)
    n_values = np.asarray(n_values, dtype=float)

    valid = np.isfinite(rmse_values) & np.isfinite(n_values) & (n_values > 0)

    if valid.sum() == 0:
        return np.nan

    return float(np.sqrt(np.sum(n_values[valid] * rmse_values[valid] ** 2) / np.sum(n_values[valid])))


def summarize_group(g):
    """
    对一个分组统计整月误差。

    输入 g 是某个变量、某个时效、某个层次下的多天结果。
    """

    n_total = int(np.nansum(g["n"].values))

    out = {
        "case_count": int(len(g)),
        "valid_case_count": int(g["n"].gt(0).sum()),
        "n_total": n_total,

        # 推荐使用的整月合成指标
        "rmse_monthly_pooled": pooled_rmse(g["rmse"], g["n"]),
        "mae_monthly_weighted": safe_weighted_mean(g["mae"], g["n"]),
        "bias_monthly_weighted": safe_weighted_mean(g["bias"], g["n"]),

        # corr 严格来说需要原始逐格数据才能精确合成；
        # 这里给出按 n 加权的近似月平均相关系数
        "corr_monthly_weighted_mean": safe_weighted_mean(g["corr"], g["n"]),

        # 日尺度指标的简单平均，便于对照
        "rmse_daily_mean": float(g["rmse"].mean(skipna=True)),
        "mae_daily_mean": float(g["mae"].mean(skipna=True)),
        "bias_daily_mean": float(g["bias"].mean(skipna=True)),
        "corr_daily_mean": float(g["corr"].mean(skipna=True)),

        # 其他辅助信息
        "pred_mean_monthly_weighted": safe_weighted_mean(g["pred_mean"], g["n"]),
        "truth_mean_monthly_weighted": safe_weighted_mean(g["truth_mean"], g["n"]),
        "diff_std_daily_mean": float(g["diff_std"].mean(skipna=True)),
    }

    return pd.Series(out)


def clean_pressure_level(df):
    """
    统一 pressure_level 字段，避免 NaN 分组丢失。
    surface 层统一写成 surface。
    upper 层保持 1000、925 等。
    """

    df = df.copy()

    if "pressure_level" not in df.columns:
        df["pressure_level"] = np.nan

    df["pressure_level_label"] = df["pressure_level"].apply(
        lambda x: "surface" if pd.isna(x) else str(int(float(x)))
    )

    return df


def filter_data(df):
    df = df.copy()

    if "target_time" in df.columns:
        df["target_time_dt"] = pd.to_datetime(
            df["target_time"],
            format="%Y-%m-%d-%H-%M",
            errors="coerce"
        )

    elif "valid_time" in df.columns:
        df["target_time_dt"] = pd.to_datetime(
            df["valid_time"],
            format="%Y-%m-%d-%H-%M",
            errors="coerce"
        )

    else:
        df["target_time_dt"] = pd.NaT

    if TARGET_MONTH is not None and "target_time_dt" in df.columns:
        month_period = pd.Period(TARGET_MONTH, freq="M")
        df = df[df["target_time_dt"].dt.to_period("M") == month_period]

    if DATA_TYPE_FILTER is not None and "data_type" in df.columns:
        df = df[df["data_type"].astype(str).str.lower() == DATA_TYPE_FILTER.lower()]

    return df


# ============================================================
# 3. 主程序
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"输入文件不存在：{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    print("=" * 100)
    print("开始统计整月平均误差")
    print(f"输入文件: {INPUT_CSV}")
    print(f"输出目录: {OUT_DIR}")
    print(f"原始行数: {len(df)}")
    print("=" * 100)

    required_cols = [
        "data_group",
        "variable",
        "n",
        "rmse",
        "bias",
        "mae",
        "corr",
        "pred_mean",
        "truth_mean",
        "diff_std",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"缺少必要字段：{col}。当前字段为：{list(df.columns)}")

    # 老版本可能叫 forecast_hour，新版本叫 lead_hour
    if "lead_hour" not in df.columns:
        if "forecast_hour" in df.columns:
            df["lead_hour"] = df["forecast_hour"]
        else:
            raise KeyError("缺少 lead_hour 或 forecast_hour 字段，无法按预报时效统计。")

    # 老版本可能没有 data_type
    if "data_type" not in df.columns:
        df["data_type"] = "unknown"

    df = clean_pressure_level(df)
    df = filter_data(df)

    print(f"筛选后行数: {len(df)}")

    if df.empty:
        raise RuntimeError("筛选后没有数据，请检查 TARGET_MONTH 或 DATA_TYPE_FILTER 设置。")

    # 数值字段强制转为数值类型
    numeric_cols = [
        "lead_hour",
        "pressure_level",
        "n",
        "rmse",
        "bias",
        "mae",
        "corr",
        "pred_mean",
        "truth_mean",
        "diff_std",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ========================================================
    # 3.1 每个变量、每个时效的整月平均误差
    # ========================================================

    group_cols = [
        "data_type",
        "data_group",
        "pressure_level_label",
        "variable",
        "lead_hour",
    ]

    monthly_by_lead_variable = (
        df
        .groupby(group_cols, dropna=False)
        .apply(summarize_group)
        .reset_index()
        .sort_values(
            ["data_type", "data_group", "variable", "pressure_level_label", "lead_hour"]
        )
    )

    monthly_by_lead_variable.to_csv(
        OUT_DIR / "monthly_by_lead_variable.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 3.2 每个变量整月总体误差，不区分预报时效
    # ========================================================

    group_cols_all_leads = [
        "data_type",
        "data_group",
        "pressure_level_label",
        "variable",
    ]

    monthly_by_variable_all_leads = (
        df
        .groupby(group_cols_all_leads, dropna=False)
        .apply(summarize_group)
        .reset_index()
        .sort_values(
            ["data_type", "data_group", "variable", "pressure_level_label"]
        )
    )

    monthly_by_variable_all_leads.to_csv(
        OUT_DIR / "monthly_by_variable_all_leads.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 3.3 近地面变量统计
    # ========================================================

    df_surface = df[df["data_group"] == "surface"].copy()

    if not df_surface.empty:
        monthly_surface_by_lead = (
            df_surface
            .groupby(["data_type", "variable", "lead_hour"], dropna=False)
            .apply(summarize_group)
            .reset_index()
            .sort_values(["data_type", "variable", "lead_hour"])
        )

        monthly_surface_by_lead.to_csv(
            OUT_DIR / "monthly_surface_by_lead.csv",
            index=False,
            encoding="utf-8-sig"
        )
    else:
        monthly_surface_by_lead = pd.DataFrame()

    # ========================================================
    # 3.4 高空变量按气压层统计
    # ========================================================

    df_upper = df[df["data_group"] == "upper"].copy()

    if not df_upper.empty:
        monthly_upper_by_level = (
            df_upper
            .groupby(
                ["data_type", "variable", "pressure_level_label", "lead_hour"],
                dropna=False
            )
            .apply(summarize_group)
            .reset_index()
            .sort_values(["data_type", "variable", "pressure_level_label", "lead_hour"])
        )

        monthly_upper_by_level.to_csv(
            OUT_DIR / "monthly_upper_by_level.csv",
            index=False,
            encoding="utf-8-sig"
        )
    else:
        monthly_upper_by_level = pd.DataFrame()

    # ========================================================
    # 3.5 输出 Excel 汇总表
    # ========================================================

    if SAVE_EXCEL:
        excel_path = OUT_DIR / "monthly_error_statistics.xlsx"

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            monthly_by_lead_variable.to_excel(
                writer,
                sheet_name="by_lead_variable",
                index=False
            )

            monthly_by_variable_all_leads.to_excel(
                writer,
                sheet_name="by_variable_all_leads",
                index=False
            )

            if not monthly_surface_by_lead.empty:
                monthly_surface_by_lead.to_excel(
                    writer,
                    sheet_name="surface_by_lead",
                    index=False
                )

            if not monthly_upper_by_level.empty:
                monthly_upper_by_level.to_excel(
                    writer,
                    sheet_name="upper_by_level",
                    index=False
                )

    # ========================================================
    # 3.6 控制台输出简要结果
    # ========================================================

    print("\n整月平均误差统计完成。")
    print("\n输出文件：")
    print(OUT_DIR / "monthly_by_lead_variable.csv")
    print(OUT_DIR / "monthly_by_variable_all_leads.csv")

    if not monthly_surface_by_lead.empty:
        print(OUT_DIR / "monthly_surface_by_lead.csv")

    if not monthly_upper_by_level.empty:
        print(OUT_DIR / "monthly_upper_by_level.csv")

    if SAVE_EXCEL:
        print(OUT_DIR / "monthly_error_statistics.xlsx")

    print("\n近地面变量整月平均误差预览：")

    preview_cols = [
        "data_type",
        "variable",
        "lead_hour",
        "case_count",
        "n_total",
        "rmse_monthly_pooled",
        "mae_monthly_weighted",
        "bias_monthly_weighted",
        "corr_monthly_weighted_mean",
    ]

    if not monthly_surface_by_lead.empty:
        print(monthly_surface_by_lead[preview_cols].to_string(index=False))

    print("=" * 100)


if __name__ == "__main__":
    main()
