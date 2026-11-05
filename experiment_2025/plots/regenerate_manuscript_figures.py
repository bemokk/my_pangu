"""Regenerate the eight manuscript figures from Statistics/data, without inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from . import (plot_metrics as metrics, plot_direction_frequency_radar as radar,
                   plot_east_china_sea_seasonal_wind_metrics as seasonal,
                   plot_wind_speed_beaufort_metrics as beaufort,
                   plot_icoads_distribution as icoads,
                   plot_cold_surge_station_wind_timeseries as cold,
                   plot_typhoon_station_wind_timeseries as typhoon,
                   plot_typhoon_tracks_stations as tracks)
    from .publication_style import FONT_FAMILY, set_style
except ImportError:
    import plot_metrics as metrics
    import plot_direction_frequency_radar as radar
    import plot_east_china_sea_seasonal_wind_metrics as seasonal
    import plot_wind_speed_beaufort_metrics as beaufort
    import plot_icoads_distribution as icoads
    import plot_cold_surge_station_wind_timeseries as cold
    import plot_typhoon_station_wind_timeseries as typhoon
    import plot_typhoon_tracks_stations as tracks
    from publication_style import FONT_FAMILY, set_style

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=ROOT/"Statistics"/"data")
    parser.add_argument("--figures-root",type=Path,default=Path(__file__).resolve().parent/"figures")
    args=parser.parse_args()
    data, output=args.data_root,args.figures_root
    output.mkdir(parents=True,exist_ok=True)
    set_style()
    outputs=[]
    audit=[]
    original_save=matplotlib.figure.Figure.savefig

    def checked_save(fig, filename, *a, **kw):
        """Check every exported figure using the actual drawn text/legend geometry."""
        from matplotlib.text import Text
        if Path(filename).suffix==".png":
            fig.canvas.draw()
            renderer=fig.canvas.get_renderer()
            legends=list(fig.legends)+[ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
            texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
            assert fig._suptitle is None, "Overall title remains"
            assert all(t.get_fontfamily()==FONT_FAMILY for t in texts), "Mixed font families"
            for legend in legends:
                box=legend.get_window_extent(renderer)
                assert fig.bbox.contains(box.x0,box.y0) and fig.bbox.contains(box.x1,box.y1), "Clipped legend"
            audit.append({"file":Path(filename).name,"font_family":FONT_FAMILY,
                          "overall_title":False,"legends":len(legends),"panels":len(fig.axes),
                          "line_points":sum(len(line.get_xdata()) for ax in fig.axes for line in ax.lines)})
        return original_save(fig,filename,*a,**kw)

    matplotlib.figure.Figure.savefig=checked_save
    try:
        outputs.append(metrics.plot_combined_metrics(metrics.load_metrics(data/"wind_speed_metrics_by_lead.csv"),
                       metrics.load_metrics(data/"wind_direction_metrics_by_lead.csv"),output)[0])
        print("1/8 lead metrics",flush=True)
        seasonal.DATA_CSV=data/"east_china_sea_wind_speed_metrics_by_season_lead_2025.csv"
        seasonal.FIGURES_DIR=output
        seasonal.main()
        outputs.append(output/"east_china_sea_wind_speed_metrics_by_season_2025.png")
        print("2/8 seasonal metrics",flush=True)
        beaufort.METRICS_CSV=data/"wind_speed_metrics_by_beaufort_2025.csv"
        beaufort.FIGURES_DIR=output
        beaufort.set_style()
        outputs.append(beaufort.plot_metrics(beaufort.load_data())[0])
        print("3/8 Beaufort metrics",flush=True)
        outputs.append(radar.plot(radar.load_frequency(data),output)[0])
        print("4/8 direction frequencies",flush=True)
        grid,summary=icoads.load_data(data)
        outputs.append(icoads.plot_spatial(grid,summary,output))
        print("5/8 ICOADS distribution",flush=True)
        outputs.append(cold.plot_figure(pd.read_csv(data/"cold_surge_2025_xiaomaidao_dachen_wind_timeseries_beijing.csv"),"beijing",output))
        print("6/8 cold surge",flush=True)
        outputs.append(typhoon.plot_figure(pd.read_csv(data/"typhoon_three_cases_chronological_beijing.csv"),
                       "beijing",output,"typhoon_three_cases_chronological"))
        print("7/8 typhoon timeseries",flush=True)
        outputs.append(tracks.plot(data,output))
        print("8/8 typhoon tracks",flush=True)
    finally:
        matplotlib.figure.Figure.savefig=original_save
    (output/"figure_style_qa.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Generated eight figures (PNG, PDF, SVG):")
    for p in outputs: print(p)


if __name__=="__main__":
    main()
