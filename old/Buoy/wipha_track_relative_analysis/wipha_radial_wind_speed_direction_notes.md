# Wipha radial wind speed and direction analysis notes

## Data and experiment definitions

- ERA5 reference: `E:\PyCharm_WorkSpace\pangu\model_input\multi_time_point\wind10_july_august.nc` at each valid time.
- GDAS_Realtime: Pangu forecast initialized from GDAS at 2025-07-17 00:00 UTC.
- ERA5_Lagged: Pangu forecast initialized from ERA5 at 2025-07-12 00:00 UTC, with a 120 h availability lag.
- Typhoon center: `E:\PyCharm_WorkSpace\pangu\ibtracs.WP.list.v04r01.csv`, WIPHA SID `2025197N14131`, using IBTrACS `LAT` and `LON` with linear time interpolation when needed.
- Land-sea mask: `E:\PyCharm_WorkSpace\pangu\src\lsm.nc`; used = `True`. Ocean is defined as lsm < 0.5.

ERA5 reference is used as a common reference field, not as independent observational truth.

## Key valid times and model steps

| lead_hour | valid_time (UTC) | GDAS model step | ERA5_Lagged model step |
|---:|---|---:|---:|
| 24 | 2025-07-18 00:00 | 24 | 144 |
| 48 | 2025-07-19 00:00 | 48 | 168 |
| 72 | 2025-07-20 00:00 | 72 | 192 |

## Methods

- All forecast u10/v10 components are linearly interpolated to the ERA5 reference grid before wind diagnostics are calculated.
- Analysis uses common valid ocean grid points within 600 km of the IBTrACS center.
- Radius bins: 0-150 km, 150-300 km, 300-600 km, and overall 0-600 km.
- Wind speed: `sqrt(u10^2 + v10^2)`.
- Meteorological wind direction: `(270 - atan2(v10, u10) * 180 / pi) % 360`.
- Signed direction error: `((model_dir - ERA5_dir + 180) % 360) - 180`.
- Direction MAE and RMSE are calculated from circular angular differences, not ordinary angle subtraction.
- Direction bias is the arithmetic mean of signed circular errors in [-180, 180] degrees.
- Winner ties: speed RMSE difference < 0.1 m/s; direction MAE difference < 2 degrees.

## Sample sizes

- T+24 h: 0-150 km: n=95, 150-300 km: n=268, 300-600 km: n=1024, 0-600 km: n=1387.
- T+48 h: 0-150 km: n=95, 150-300 km: n=272, 300-600 km: n=979, 0-600 km: n=1346.
- T+72 h: 0-150 km: n=87, 150-300 km: n=196, 300-600 km: n=676, 0-600 km: n=959.

## Outputs

- `wipha_track_relative_analysis\table_wipha_radial_wind_speed_direction_statistics.csv`
- `wipha_track_relative_analysis\table_wipha_key_lead_error_summary_for_paper.csv`
- `wipha_track_relative_analysis\supplementary_wipha_0_72h_error_statistics.csv`
- `wipha_track_relative_analysis\figure_wipha_radial_wind_speed_direction_structure.png`
- `wipha_track_relative_analysis\figure_wipha_radial_wind_speed_direction_structure.svg`
- `wipha_track_relative_analysis\figure_wipha_wind_speed_direction_error_timeseries.png`
- `wipha_track_relative_analysis\figure_wipha_wind_speed_direction_error_timeseries.svg`
- `wipha_track_relative_analysis\figure_wipha_radial_wind_speed_mae_timeseries.png`
- `wipha_track_relative_analysis\figure_wipha_radial_wind_speed_mae_timeseries.svg`

The optional 0-72 h time-series figure was generated: `True`.

## Interpretation limits

- Results quantify agreement with the ERA5 reference field and are not an independent observational validation.
- Grid-point statistics are spatially correlated and should not be interpreted as independent samples.
- Near-coastal results depend on the 0.25-degree land-sea mask and interpolation behavior.
