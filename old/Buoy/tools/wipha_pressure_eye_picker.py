from __future__ import annotations

import argparse
import sys
import tkinter as tk
from datetime import timedelta
from pathlib import Path
from tkinter import filedialog, ttk

BUOY_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BUOY_DIR.parent
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from plots.plot_wipha_pressure_eye_check import (
    DATASET_LABELS,
    DEFAULT_SCHEMES,
    OUT_ROOT,
    PLOT_BOX,
    TRACK_INIT,
    WIPHA_SEARCH_BOX,
    _official_track_by_lead,
    _pressure_levels,
    apply_manual_eye_override,
    load_manual_eye_overrides,
    locate_center_from_msl,
    nearest_msl_hpa_at_point,
    normalize_scheme,
    pressure_eye_check_leads,
    read_surface_msl,
    subset_msl_hpa,
    surface_array_path,
    upsert_manual_eye_override,
)
from plots.wipha_case_common import moving_box


def default_manual_eye_csv_path() -> Path:
    return OUT_ROOT / "manual_eye_overrides.csv"


def scheme_options() -> list[tuple[str, str]]:
    return [(scheme, DATASET_LABELS.get(scheme, scheme)) for scheme in DEFAULT_SCHEMES]


def lead_options() -> list[tuple[int, str]]:
    rows = []
    for lead_hour in pressure_eye_check_leads():
        valid_time = TRACK_INIT + timedelta(hours=lead_hour)
        rows.append((lead_hour, f"+{lead_hour:02d} h / {valid_time:%Y-%m-%d %H} UTC"))
    return rows


def format_click_status(scheme: str, lead_hour: int, lon: float, lat: float, csv_path: Path) -> str:
    label = DATASET_LABELS.get(normalize_scheme(scheme), scheme)
    return f"{label} +{lead_hour:02d} h saved: {lon:.3f}E, {lat:.3f}N -> {csv_path}"


class PressureEyePickerApp:
    def __init__(self, root: tk.Tk, manual_eye_csv: Path | None = None, scheme: str = "era5_lagged_5d", lead_hour: int = 0):
        self.root = root
        self.manual_eye_csv = Path(manual_eye_csv) if manual_eye_csv else default_manual_eye_csv_path()
        self.official_by_lead = _official_track_by_lead()
        self.current_msl: np.ndarray | None = None
        self.scheme_values = [item[0] for item in scheme_options()]
        self.scheme_labels = {value: label for value, label in scheme_options()}
        self.lead_values = [item[0] for item in lead_options()]
        self.lead_labels = {value: label for value, label in lead_options()}

        self.root.title("Wipha Pressure Eye Picker")
        self.root.geometry("1120x820")

        controls = ttk.Frame(root, padding=6)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Experiment").pack(side=tk.LEFT, padx=(0, 4))
        self.scheme_var = tk.StringVar(value=normalize_scheme(scheme))
        self.scheme_combo = ttk.Combobox(
            controls,
            state="readonly",
            width=30,
            values=[self.scheme_labels[value] for value in self.scheme_values],
        )
        self.scheme_combo.current(max(0, self.scheme_values.index(normalize_scheme(scheme))))
        self.scheme_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.scheme_combo.bind("<<ComboboxSelected>>", lambda _event: self.reload_plot())

        ttk.Label(controls, text="Lead").pack(side=tk.LEFT, padx=(0, 4))
        self.lead_combo = ttk.Combobox(
            controls,
            state="readonly",
            width=28,
            values=[self.lead_labels[value] for value in self.lead_values],
        )
        self.lead_combo.current(max(0, self.lead_values.index(int(lead_hour))))
        self.lead_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.lead_combo.bind("<<ComboboxSelected>>", lambda _event: self.reload_plot())

        ttk.Button(controls, text="Previous", command=self.previous_lead).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Next", command=self.next_lead).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Reload", command=self.reload_plot).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="CSV", command=self.choose_csv).pack(side=tk.LEFT, padx=2)

        self.figure = Figure(figsize=(9.2, 6.6), dpi=120)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        toolbar = NavigationToolbar2Tk(self.canvas, root, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)

        self.status_var = tk.StringVar(value=f"Manual CSV: {self.manual_eye_csv}")
        ttk.Label(root, textvariable=self.status_var, padding=6, wraplength=1080).pack(side=tk.BOTTOM, fill=tk.X)

        self.reload_plot()

    def selected_scheme(self) -> str:
        index = self.scheme_combo.current()
        return self.scheme_values[index]

    def selected_lead_hour(self) -> int:
        index = self.lead_combo.current()
        return self.lead_values[index]

    def previous_lead(self) -> None:
        index = self.lead_combo.current()
        if index > 0:
            self.lead_combo.current(index - 1)
            self.reload_plot()

    def next_lead(self) -> None:
        index = self.lead_combo.current()
        if index < len(self.lead_values) - 1:
            self.lead_combo.current(index + 1)
            self.reload_plot()

    def choose_csv(self) -> None:
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Select manual eye override CSV",
            initialfile=self.manual_eye_csv.name,
            initialdir=str(self.manual_eye_csv.parent),
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if filename:
            self.manual_eye_csv = Path(filename)
            self.reload_plot()

    def load_model_eye_for_lead(self, scheme: str, lead_hour: int) -> dict:
        last_center: tuple[float, float] | None = None
        for item_lead in pressure_eye_check_leads():
            valid_time = TRACK_INIT + timedelta(hours=item_lead)
            path = surface_array_path(scheme, item_lead, valid_time)
            msl = read_surface_msl(path)
            search_box = WIPHA_SEARCH_BOX if last_center is None else moving_box(*last_center)
            model_eye = locate_center_from_msl(msl, search_box)
            last_center = (model_eye["center_lon"], model_eye["center_lat"])
            if item_lead == lead_hour:
                self.current_msl = msl
                return model_eye
        raise RuntimeError(f"No pressure field could be loaded for {scheme} +{lead_hour}h")

    def reload_plot(self) -> None:
        scheme = self.selected_scheme()
        lead_hour = self.selected_lead_hour()
        valid_time = pd.Timestamp(TRACK_INIT + timedelta(hours=lead_hour))
        try:
            auto_eye = self.load_model_eye_for_lead(scheme, lead_hour)
            if self.current_msl is None:
                raise RuntimeError("Pressure field was not loaded.")
            manual_overrides = load_manual_eye_overrides(self.manual_eye_csv) if self.manual_eye_csv.exists() else {}
            manual_msl_hpa = None
            if (scheme, lead_hour) in manual_overrides:
                manual_msl_hpa = nearest_msl_hpa_at_point(
                    self.current_msl,
                    manual_overrides[(scheme, lead_hour)]["lon"],
                    manual_overrides[(scheme, lead_hour)]["lat"],
                )
            model_eye = apply_manual_eye_override(
                auto_eye,
                manual_overrides,
                scheme=scheme,
                lead_hour=lead_hour,
                manual_msl_hpa=manual_msl_hpa,
            )
            msl_hpa, lats, lons = subset_msl_hpa(self.current_msl)
            official_eye = self.official_by_lead.loc[lead_hour] if lead_hour in self.official_by_lead.index else None
            self.draw_pressure_map(scheme, lead_hour, valid_time, msl_hpa, lats, lons, auto_eye, model_eye, official_eye)
            self.status_var.set(f"Click the map to save a manual eye position. Manual CSV: {self.manual_eye_csv}")
        except Exception as exc:
            self.figure.clear()
            self.canvas.draw_idle()
            self.status_var.set(f"Could not load pressure field: {exc}")

    def draw_pressure_map(
        self,
        scheme: str,
        lead_hour: int,
        valid_time: pd.Timestamp,
        msl_hpa: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        auto_eye: dict,
        model_eye: dict,
        official_eye: pd.Series | None,
    ) -> None:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        from land_mask import load_land_union

        self.figure.clear()
        projection = ccrs.PlateCarree()
        ax = self.figure.add_subplot(111, projection=projection)
        ax.set_extent([PLOT_BOX[0], PLOT_BOX[1], PLOT_BOX[2], PLOT_BOX[3]], crs=projection)
        ax.set_facecolor("#EAF3F8")

        land_union = load_land_union(PLOT_BOX[0], PLOT_BOX[2], PLOT_BOX[1], PLOT_BOX[3])
        ax.add_geometries([land_union], crs=projection, facecolor="#D7D2C3", edgecolor="#777777", linewidth=0.35, zorder=3)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.45, edgecolor="#333333", zorder=4)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.25, edgecolor="#777777", zorder=4)

        levels = _pressure_levels(msl_hpa)
        mesh = ax.contourf(lons, lats, msl_hpa, levels=levels, cmap="Spectral_r", extend="both", transform=projection, zorder=1)
        contour_step = 4.0 if len(levels) > 20 else 2.0
        contour_levels = np.arange(np.nanmin(levels), np.nanmax(levels) + contour_step, contour_step)
        contours = ax.contour(
            lons,
            lats,
            msl_hpa,
            levels=contour_levels,
            colors="#4A4A4A",
            linewidths=0.42,
            alpha=0.72,
            transform=projection,
            zorder=2,
        )
        ax.clabel(contours, inline=True, fmt="%.0f", fontsize=6.4)

        ax.scatter(auto_eye["center_lon"], auto_eye["center_lat"], marker="+", s=110, color="#555555", linewidth=1.6, label="Auto eye", transform=projection, zorder=6)
        ax.scatter(
            model_eye["center_lon"],
            model_eye["center_lat"],
            marker="*",
            s=150,
            color="#D62728",
            edgecolor="white",
            linewidth=0.75,
            label="Selected eye" if model_eye.get("manual_override") else "Model eye",
            transform=projection,
            zorder=7,
        )
        if official_eye is not None and pd.notna(official_eye.get("lon")) and pd.notna(official_eye.get("lat")):
            ax.scatter(official_eye["lon"], official_eye["lat"], marker="X", s=80, color="#111111", edgecolor="white", linewidth=0.65, label="Official eye", transform=projection, zorder=8)

        gl = ax.gridlines(crs=projection, draw_labels=True, linewidth=0.32, color="#777777", alpha=0.38, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        colorbar = self.figure.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.82, pad=0.035)
        colorbar.set_label("Mean sea-level pressure (hPa)")
        ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
        ax.set_title(
            f"{DATASET_LABELS.get(scheme, scheme)} MSLP eye picker    Valid {valid_time:%Y-%m-%d %H:%M UTC}    +{lead_hour:02d} h",
            loc="left",
            fontweight="bold",
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def on_canvas_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        lon = float(event.xdata)
        lat = float(event.ydata)
        scheme = self.selected_scheme()
        lead_hour = self.selected_lead_hour()
        try:
            upsert_manual_eye_override(self.manual_eye_csv, scheme, lead_hour, lon, lat)
            self.status_var.set(format_click_status(scheme, lead_hour, lon, lat, self.manual_eye_csv))
            self.reload_plot()
        except Exception as exc:
            self.status_var.set(f"Could not save manual eye position: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick Typhoon Wipha model eye positions from MSLP fields.")
    parser.add_argument("--manual-eye-csv", type=Path, default=default_manual_eye_csv_path())
    parser.add_argument("--scheme", default="era5_lagged_5d", choices=["gdas", "gdas_forecast", "era5", "era5_lagged", "era5_lagged_5d"])
    parser.add_argument("--lead", type=int, default=0, choices=pressure_eye_check_leads())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    PressureEyePickerApp(root, args.manual_eye_csv, args.scheme, args.lead)
    root.mainloop()


if __name__ == "__main__":
    main()
