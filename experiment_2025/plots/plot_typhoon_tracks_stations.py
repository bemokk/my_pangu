"""Plot the three JMA best tracks and their selected coastal stations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe
import numpy as np

try:
    from .publication_style import set_style, format_figure
except ImportError:
    from publication_style import set_style, format_figure

ROOT = Path(__file__).resolve().parents[1]
AREA = [35.0, 98.0, 8.0, 150.0]  # lat_max, lon_min, lat_min, lon_max
COLORS = ["#59C878", "#E978B1", "#A07AEC"]
LINESTYLES = ["-", "--", ":"]


def mercator_y(latitude):
    return np.degrees(np.log(np.tan(np.pi / 4 + np.radians(latitude) / 2)))


def plot(data_root: Path, figures_root: Path, *, ax=None):
    set_style()
    tracks = json.loads((data_root / "typhoon_tracks_jma_2025.json").read_text(encoding="utf-8"))["tracks"]
    stations = {d["code"]: d for d in json.loads((data_root / "typhoon_map_stations.json").read_text(encoding="utf-8"))}
    geometry = json.loads((data_root / "typhoon_map_geometry.geojson").read_text(encoding="utf-8"))
    embedded = ax is not None
    if not embedded:
        fig, ax = plt.subplots(figsize=(12, 7.30))
    else:
        fig = ax.figure
    for feature in geometry["features"]:
        geom = feature["geometry"]
        polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for polygon in polygons:
            points = np.asarray(polygon[0])
            ax.add_patch(Polygon(np.column_stack((points[:, 0], mercator_y(points[:, 1]))),
                                 facecolor="#E8E8E8", edgecolor="#D1D1D1", linewidth=.35, zorder=2))
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    for i, track in enumerate(tracks):
        points = track["points"]
        lon = np.array([p["lon"] for p in points])
        y = mercator_y([p["lat"] for p in points])
        ax.plot(lon, y, color=COLORS[i], linestyle=LINESTYLES[i], linewidth=1.8,
                marker="o", markersize=3, zorder=3)
        for j in range(4, len(points), 6):
            ax.annotate("", xy=(lon[j], y[j]), xytext=(lon[j-1], y[j-1]),
                        arrowprops=dict(arrowstyle="-|>", color=COLORS[i], lw=1.5, mutation_scale=14), zorder=4)
        target = next(j for j, p in enumerate(points) if p["time"] == track["target"])
        ax.plot(lon[target], y[target], marker="s", markersize=8, markerfacecolor="white",
                markeredgecolor=COLORS[i], markeredgewidth=1.8, zorder=5)
        ax.annotate(track["target"][5:10].replace("-", "/") + " 00 UTC",
                    (lon[target], y[target]), xytext=((-7 if i==0 else 8), (10 if i==1 else -16)),
                    textcoords="offset points", ha="right" if i==0 else "left", fontsize=10,
                    path_effects=halo, zorder=6)
        ax.annotate(track["name"]+" "+track["id"], (lon[0],y[0]),
                    xytext=(-8 if i==2 else 8, -15 if i!=1 else 12), textcoords="offset points",
                    ha="right" if i==2 else "left", fontsize=11, path_effects=halo, zorder=6)
        station = stations[track["station"]]
        sx, sy = station["lon"], mercator_y(station["lat"])
        ax.scatter([sx], [sy], s=42, color="#329AF0", zorder=7)
        left = track["station"] in ("DSN", "ZLG")
        ax.annotate(station["name"], (sx, sy), xytext=(-10 if left else 10, -2),
                    textcoords="offset points", ha="right" if left else "left", va="center",
                    fontsize=11, path_effects=halo, zorder=8)
    for label, lon, lat in [("东海",125.5,29.1),("南海",117,16.2),("台湾岛",123.9,23.8)]:
        ax.text(lon,mercator_y(lat),label,ha="center",color="#888888",fontsize=11,zorder=3)
    ax.set_xlim(AREA[1], AREA[3])
    ax.set_ylim(mercator_y(AREA[2]), mercator_y(AREA[0]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([98,110,120,130,140,150])
    ax.set_xticklabels([f"{v}°E" for v in [98,110,120,130,140,150]])
    lat_ticks=[8,10,15,20,25,30,35]
    ax.set_yticks(mercator_y(lat_ticks))
    ax.set_yticklabels([f"{v}°N" for v in lat_ticks])
    ax.tick_params(length=0, pad=6, labelsize=10)
    ax.grid(color="#DEDEDE", linewidth=.5, zorder=1)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(.8)
    handles=[Line2D([],[],ls="",marker="o",color="#329AF0",label="海洋站")]
    handles += [Line2D([],[],color=COLORS[i],ls=LINESTYLES[i],lw=1.8,label=t["name"]+" "+t["id"]) for i,t in enumerate(tracks)]
    handles += [Line2D([],[],ls="",marker="s",mfc="white",mec="black",label="固定起报时刻（UTC）")]
    if not embedded:
        fig.subplots_adjust(left=.055,right=.985,top=.985,bottom=.065)
        format_figure(fig)
    ax.legend(handles=handles,loc="upper left",ncol=1,fontsize=10,frameon=True,
              facecolor="white",edgecolor="black",framealpha=1,fancybox=False)
    if embedded:
        return ax
    figures_root.mkdir(parents=True,exist_ok=True)
    stem=figures_root/"typhoon_three_cases_tracks_stations"
    for suffix in ("png","pdf","svg"):
        fig.savefig(stem.with_suffix("."+suffix),dpi=600)
    plt.close(fig)
    return stem.with_suffix(".png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=ROOT/"Statistics"/"data")
    parser.add_argument("--figures-root",type=Path,default=Path(__file__).resolve().parent/"figures")
    args=parser.parse_args()
    print(plot(args.data_root,args.figures_root))


if __name__=="__main__":
    main()
