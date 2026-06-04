# Wipha ICOADS Verification Plotting Notes

Matched sample file: `E:\PyCharm_WorkSpace\pangu\Buoy\results\wipha_track_relative_analysis\data\matched_icoads_model_samples.csv`

ICOADS observations are used as the verification reference.

## Filtering

- Kept samples with distance_to_typhoon_center_km <= 600.
- Kept lead_hour_from_case_start within 0-72 h.
- Kept samples where obs_wind_speed, gdas_wind_speed, and era5lagged_wind_speed are non-missing.
- Kept wind speeds within 0-75 m/s.
- Filtered sample count: 59.
- Samples before 18 h: 1. Missing lead hours: [3, 6, 9, 12, 15, 21, 24, 27, 30, 33].

## Sample Counts

- 0-400 km: 19.
- 400-600 km: 40.
- 18-48 h: 11.
- 48-72 h: 47.
- 18-72 h: 58.

## Figures

- figure_icoads_sample_coverage: spatial coverage of filtered ICOADS samples, colored by lead time and sized by observed wind speed.
- figure_sample_count_distribution: sample counts by lead hour, coarse radius bin, and forecast period.
- figure_wind_speed_error_overall: overall GDAS_Realtime and ERA5_Lagged wind speed errors against ICOADS.
- figure_wind_error_by_period: period-aggregated RMSE and MAE, with sample counts marked on each bar.

The best track from typhoon_2506_Wipha.csv was overlaid on the sample coverage map.
Cartopy was not available in the rendering environment, so the sample coverage map uses ordinary longitude-latitude axes.

Due to the limited number and uneven temporal-spatial distribution of ICOADS samples, the figures support point-based verification only and should not be interpreted as a full validation of the typhoon inner-core wind structure.
