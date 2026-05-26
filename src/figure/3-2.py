# -*- coding: utf-8 -*-
"""
plot_figure3_2_upper_rmse.py

功能：
绘制图3-2：高空代表性动力变量 RMSE 随预报时效变化曲线

子图包括：
(a) 500 hPa Z RMSE
(b) 850 hPa U RMSE
(c) 850 hPa V RMSE
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 用户配置区
# ============================================================

GDAS_CSV = Path(r"E:\pyCharmProject\pangu\src\comparison_results\gdas_monthly_by_lead_variable.csv")
ERA5_CSV = Path(r"E:\pyCharmProject\pangu\src\comparison_results\era5_monthly_by_lead_variable.csv")

OUT_DIR = Path(r"./figure3_1_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEAD_ORDER = [1, 3, 6, 12, 24, 48, 72]

# 图3-2的三个子图配置 (移除了单独的 color 配置)
PLOT_ITEMS = [
    {
        "var": "z",
        "level": 500,
        "title": "(a) 500 hPa Z RMSE",
        "ylabel": "RMSE",
    },
    {
        "var": "u",
        "level": 850,
        "title": "(b) 850 hPa U RMSE",
        "ylabel": "RMSE (m s$^{-1}$)",
    },
    {
        "var": "v",
        "level": 850,
        "title": "(c) 850 hPa V RMSE",
        "ylabel": "RMSE (m s$^{-1}$)",
    },
]


# ============================================================
# 2. 数据读取与筛选
# ============================================================

def load_monthly_csv(csv_path: Path, scheme_name: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"文件不存在：{csv_path}")

    df = pd.read_csv(csv_path)
    df["scheme"] = scheme_name

    df["lead_hour"] = pd.to_numeric(df["lead_hour"], errors="coerce")
    df["pressure_level_num"] = pd.to_numeric(df["pressure_level_label"], errors="coerce")

    return df


def select_upper_rmse(df: pd.DataFrame, var_name: str, pressure_level: int) -> pd.DataFrame:
    out = df[
        (df["data_group"] == "upper") &
        (df["variable"] == var_name) &
        (df["pressure_level_num"] == pressure_level)
    ].copy()

    out = out.sort_values("lead_hour")

    if out.empty:
        raise ValueError(
            f"没有找到变量 {var_name}、气压层 {pressure_level} hPa 的数据。"
            f"请检查 pressure_level_label 和 variable 字段。"
        )

    return out


df_gdas = load_monthly_csv(GDAS_CSV, "GDAS_Realtime")
df_era5 = load_monthly_csv(ERA5_CSV, "ERA5_Lagged")


# ============================================================
# 3. 绘图风格
# ============================================================

def set_plot_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


def style_axis(ax):
    ax.set_facecolor("#EAEAF2")
    ax.grid(True, color="white", linewidth=1.2)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks(LEAD_ORDER)
    ax.set_xticklabels([f"{x}" for x in LEAD_ORDER])


# ============================================================
# 4. 单个子图绘制
# ============================================================

def plot_one_panel(ax, item):
    var_name = item["var"]
    level = item["level"]

    g = select_upper_rmse(df_gdas, var_name, level)
    e = select_upper_rmse(df_era5, var_name, level)

    # 统一指定两套方案的颜色
    color_gdas = "#4C72B0"  # 蓝色
    color_era5 = "#DD8452"  # 橙色

    # GDAS_Realtime (全蓝)
    ax.plot(
        g["lead_hour"],
        g["rmse_monthly_pooled"],
        color=color_gdas,
        linestyle="-",
        linewidth=3.0,
        marker="o",
        markersize=6,
        label="GDAS_Realtime"
    )

    # ERA5_Lagged (全橙)
    ax.plot(
        e["lead_hour"],
        e["rmse_monthly_pooled"],
        color=color_era5,
        linestyle="--",
        linewidth=3.0,
        marker="s",
        markersize=6,
        label="ERA5_Lagged"
    )

    ax.set_title(item["title"], fontweight="bold")
    ax.set_xlabel("Forecast lead time (h)")
    ax.set_ylabel(item["ylabel"])

    style_axis(ax)


# ============================================================
# 5. 主绘图函数
# ============================================================

def make_figure_3_2():
    set_plot_style()

    fig, axes = plt.subplots(
        3, 1,
        figsize=(5.5, 10),
        constrained_layout=True
    )

    for ax, item in zip(axes, PLOT_ITEMS):
        plot_one_panel(ax, item)

    # 因为每个子图里已经是一蓝一橙了，直接获取第一个子图的句柄即可生成正确的图例
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04)
    )

    fig.suptitle(
        "Figure 3-2  RMSE variations of representative upper-air dynamic variables",
        fontsize=13,
        fontweight="bold",
        y=1.07
    )

    png_path = OUT_DIR / "Figure3_2_upper_dynamic_RMSE.png"
    svg_path = OUT_DIR / "Figure3_2_upper_dynamic_RMSE.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"图3-2 PNG 已保存：{png_path}")
    print(f"图3-2 SVG 已保存：{svg_path}")


# ============================================================
# 6. 运行
# ============================================================

if __name__ == "__main__":
    make_figure_3_2()