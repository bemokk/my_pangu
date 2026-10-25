# -*- coding: utf-8 -*-
"""
plot_figure3_1_rmse_only.py

功能：
绘制图3-1：近地层主要变量（MSL、U10、V10）的 RMSE 随预报时效变化曲线
子图包括：
(a) MSL RMSE
(b) U10 RMSE
(c) V10 RMSE
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent

# ============================================================
# 1. 用户配置区
# ============================================================

# 你的两个输入文件
GDAS_CSV = PROJECT_ROOT / "src" / "comparison_results" / "gdas_monthly_by_lead_variable.csv"
ERA5_CSV = PROJECT_ROOT / "src" / "comparison_results" / "era5_monthly_by_lead_variable.csv"

# 输出目录
OUT_DIR = FIGURE_DIR / "figure3_1_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 要绘制的变量
PLOT_VARS = ["msl", "u10", "v10"]

# 预报时效顺序
LEAD_ORDER = [1, 3, 6, 12, 24, 48, 72]

# 子图标题
SUBPLOT_TITLES = {
    "msl": "(a) MSL RMSE",
    "u10": "(b) U10 RMSE",
    "v10": "(c) V10 RMSE",
}

# y轴标签
Y_LABELS = {
    "msl": "RMSE (Pa)",
    "u10": "RMSE (m s$^{-1}$)",
    "v10": "RMSE (m s$^{-1}$)",
}

# 已移除按变量分类的 COLORS 字典，统一在绘图函数中设置蓝橙配色


# ============================================================
# 2. 读取并整理数据
# ============================================================

def load_surface_rmse(csv_path: Path, scheme_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 仅保留 near-surface / surface / 需要的变量
    df = df[
        (df["data_group"] == "surface") &
        (df["pressure_level_label"].astype(str) == "surface") &
        (df["variable"].isin(PLOT_VARS))
    ].copy()

    df["lead_hour"] = pd.to_numeric(df["lead_hour"], errors="coerce")
    df["scheme"] = scheme_name

    return df


gdas_df = load_surface_rmse(GDAS_CSV, "GDAS_Realtime")
era5_df = load_surface_rmse(ERA5_CSV, "ERA5_Lagged")


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
    # 背景和网格风格
    ax.set_facecolor("#EAEAF2")
    ax.grid(True, color="white", linewidth=1.2)
    ax.set_axisbelow(True)

    # 去掉上右边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # x轴
    ax.set_xticks(LEAD_ORDER)
    ax.set_xticklabels([f"{x}" for x in LEAD_ORDER])


# ============================================================
# 4. 单个子图绘制
# ============================================================

def plot_one_var(ax, var_name: str):
    g = gdas_df[gdas_df["variable"] == var_name].sort_values("lead_hour")
    e = era5_df[era5_df["variable"] == var_name].sort_values("lead_hour")

    # 统一指定两套方案的颜色
    color_gdas = "#4C72B0"  # 蓝色
    color_era5 = "#DD8452"  # 橙色

    # GDAS_Realtime：实线 (统一蓝色)
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

    # ERA5_Lagged：虚线 (统一橙色)
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

    ax.set_title(SUBPLOT_TITLES[var_name], fontweight="bold")
    ax.set_ylabel(Y_LABELS[var_name])
    ax.set_xlabel("Forecast lead time (h)")

    style_axis(ax)


# ============================================================
# 5. 主绘图函数
# ============================================================

def make_figure():
    set_plot_style()

    # 1. 改为 3行1列，调整 figsize，宽度约为 4~5 英寸，高度按比例增加
    # sharex=True 可以让三个图共享横坐标，只在最下面显示刻度，让图更紧凑
    fig, axes = plt.subplots(
        3, 1,
        figsize=(5.5, 10),
        constrained_layout=True,
        sharex=True
    )

    for ax, var_name in zip(axes, PLOT_VARS):
        plot_one_var(ax, var_name)
        # 如果使用了 sharex=True，建议把前两个子图的 x 轴标签去掉，只留最下面的
        if var_name != "v10":
            ax.set_xlabel("")

    # 2. 调整图例位置，放在整张图的上方
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04) # 微调 y 轴位置防止和标题重叠
    )

    # 3. 调整总标题位置
    fig.suptitle(
        "Figure 3-1  RMSE variations of near-surface variables",
        fontsize=13,
        fontweight="bold",
        y=1.07
    )

    png_path = OUT_DIR / "Figure3_1_RMSE_MSL_U10_V10.png"
    svg_path = OUT_DIR / "Figure3_1_RMSE_MSL_U10_V10.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"图已保存：{png_path}")
    print(f"图已保存：{svg_path}")


# ============================================================
# 6. 运行
# ============================================================

if __name__ == "__main__":
    make_figure()
