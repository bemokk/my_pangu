# Wipha 2025 Buoy and Typhoon Track Case Analysis Design

## Goal

Replicate the case-study style of Section 3.4 in `HRCLDAS-V1.0和ERA5海面风场对比评估分析.pdf` for Typhoon Wipha (2025 No. 6). The outputs should include two selected high-quality buoy/platform cases, wind speed and wind direction time series, statistical verification tables, and a 72 h typhoon track error comparison between GDAS and ERA5 lagged forecasts.

## Time Window

Use UTC time from `2025-07-17 00:00` to `2025-07-22 23:00` for buoy wind analysis.

For typhoon forecast track verification, use initialization `2025-07-17 00:00 UTC` and leads `0, 3, 6, ..., 72 h`.

## Data Sources

### Buoy/model matched samples

Use:

`Buoy/results/wind_model_statistics/wind_model_statistics_3_72h/matched_buoy_model_wind_samples.csv`

Required columns include:

- `datetime_utc`
- `platform_id`
- `latitude`, `longitude`
- `obs_speed_ms`, `obs_dir_deg`
- `dataset`
- `lead_hour`
- `pred_speed_ms`, `pred_dir_deg`
- error columns if present

### Forecast datasets for buoy time series

Use two forecast datasets:

- `gdas_forecast`
- `era5_lagged_5d`

Do not include `era5_realtime` in the main Wipha case figures/tables unless requested later.

### Typhoon best track / real track

Use the NMC typhoon JSON workflow in `trash/get_typhoon_path.py`, adapted for Typhoon Wipha 2025 No. 6. Cache downloaded path data under `Buoy/results/wipha_case/` so later plotting can proceed offline if the web endpoint is unavailable.

### Forecast fields for typhoon path extraction

GDAS 72 h forecast path:

- `model_output/gdas/2025-07-17-00-00/{lead}/output_surface_<valid_time>.npy|nc`

ERA5 lagged forecast path:

- `model_output/era5/2025-07-12-00-00/{120+lead}/output_surface_<valid_time>.npy|nc`

If `.nc` opening is unavailable because netCDF dependencies are missing, read the corresponding `.npy` arrays directly. Surface variable order is `[msl, u10, v10, t2m]`, with latitude `90..-90` every 0.25 deg and longitude `0.125..359.875` every 0.25 deg.

## Buoy Selection

Default selected platforms:

- `EVH28KM`
- `3FOS8`

Rationale from the approved screening:

- `EVH28KM`: highest combined quality score in the Wipha area, 33/48 time slots, max observed speed 16.0 m/s.
- `3FOS8`: strong wind signal and smaller movement than the strongest-drift alternatives, 16/48 time slots, max observed speed 19.5 m/s.

Treat these as high-quality platform observations in the Wipha-affected sea area. Because they are not perfectly fixed stations, the location figure should draw all observed positions and mark each platform's mean position.

## Processing Rules

### Observation aggregation

For each selected platform and observation time, aggregate duplicated observation records by mean for wind speed and circular mean for wind direction.

### Forecast value selection for time series

For each platform, time, and dataset, select the shortest available `lead_hour` record. If duplicates remain, average wind speed and use circular mean for wind direction.

This prevents mixing multiple forecast lead times at the same valid time in the time-series figure.

### Direction errors

Use circular angular differences in degrees:

`diff = ((pred - obs + 180) % 360) - 180`

Absolute direction error is `abs(diff)`, bounded in `0..180`.

## Outputs

### Figure 10 style: Wipha partial track and selected platform locations

Files:

- `Buoy/results/figures/wipha_track_buoy_locations.png`
- `Buoy/results/figures/wipha_track_buoy_locations.svg`

Contents:

- Wipha real track for the case period.
- Observed positions of `EVH28KM` and `3FOS8` during the window.
- Mean position markers and labels for both platforms.
- Map area about `105-130E, 10-32N`.

### Figure 11 style: wind speed and wind direction time series

Files:

- `Buoy/results/figures/wipha_buoy_wind_timeseries.png`
- `Buoy/results/figures/wipha_buoy_wind_timeseries.svg`

Layout:

- 2 columns: one per platform.
- 2 rows: wind speed and wind direction.

Curves:

- Buoy observation.
- GDAS forecast.
- ERA5 lagged 5d forecast.

### Table 5 style: verification statistics

Files:

- `Buoy/results/wipha_case/wipha_buoy_wind_statistics.csv`
- `Buoy/results/wipha_case/wipha_buoy_wind_statistics.xlsx` when supported.
- `Buoy/results/figures/wipha_buoy_wind_statistics_table.png`
- `Buoy/results/figures/wipha_buoy_wind_statistics_table.svg`

Statistics:

- `platform_id`
- `dataset`
- `sample_count`
- `speed_bias`
- `speed_mae`
- `speed_rmse`
- `speed_corr`
- `direction_bias`
- `direction_mae`
- `direction_rmse`
- optional `direction_corr` if meaningful

Also include combined rows across both platforms.

### Track forecast and error comparison

Files:

- `Buoy/results/wipha_case/wipha_typhoon_tracks_2025071700.csv`
- `Buoy/results/wipha_case/wipha_typhoon_track_errors_2025071700.csv`
- `Buoy/results/figures/wipha_track_forecast_error_2025071700.png`
- `Buoy/results/figures/wipha_track_forecast_error_2025071700.svg`

Contents:

- Real Wipha track interpolated to forecast valid times.
- GDAS forecast track from `2025-07-17 00 UTC`.
- ERA5 lagged forecast track from `2025-07-12 00 UTC`, valid from `2025-07-17 00 UTC`.
- Great-circle track errors in km for each lead.

## Validation

Run syntax checks for new scripts with `py_compile`.

Add or update lightweight tests for:

- Circular direction difference.
- Platform aggregation/shortest-lead selection.
- Great-circle distance sanity.

Generated binary figures are not committed unless repository convention requires them; CSV/SVG/PNG outputs may be left in the working tree for user review.
