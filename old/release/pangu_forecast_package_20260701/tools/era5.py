"""ERA5 download and Pangu input preparation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "model_input" / "single_time_point" / "era5"
SURFACE_VARIABLES = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
]
UPPER_VARIABLES = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]
PRESSURE_LEVELS = [
    "1000", "925", "850", "700", "600", "500", "400",
    "300", "250", "200", "150", "100", "50",
]


def _read_array(dataset, name: str, expected_shape: tuple[int, ...]):
    import numpy as np

    data = np.asarray(dataset.variables[name][:], dtype=np.float32).squeeze()
    if data.shape != expected_shape:
        raise ValueError(f"{name} shape is {data.shape}; expected {expected_shape}")
    return data


def download_and_prepare(date_time: datetime, output_root: Path | None = None) -> Path:
    """Download one ERA5 analysis time and write the two Pangu NPY inputs."""
    import cdsapi
    import netCDF4 as nc
    import numpy as np

    output_root = Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
    output_dir = output_root / date_time.strftime("%Y-%m-%d-%H-%M")
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_path = output_dir / "surface.nc"
    upper_path = output_dir / "upper.nc"
    client = cdsapi.Client()
    common = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "date": date_time.strftime("%Y-%m-%d"),
        "time": date_time.strftime("%H:%M"),
        "area": [90, 0, -90, 359.75],
    }

    print(f"[ERA5] downloading {date_time:%Y-%m-%d %H:%M}")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {**common, "variable": SURFACE_VARIABLES},
        str(surface_path),
    )
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {**common, "variable": UPPER_VARIABLES, "pressure_level": PRESSURE_LEVELS},
        str(upper_path),
    )

    surface = np.empty((4, 721, 1440), dtype=np.float32)
    with nc.Dataset(surface_path) as dataset:
        for index, name in enumerate(["msl", "u10", "v10", "t2m"]):
            surface[index] = _read_array(dataset, name, (721, 1440))
    np.save(output_dir / "input_surface.npy", surface)

    upper = np.empty((5, 13, 721, 1440), dtype=np.float32)
    with nc.Dataset(upper_path) as dataset:
        for index, name in enumerate(["z", "q", "t", "u", "v"]):
            upper[index] = _read_array(dataset, name, (13, 721, 1440))
    np.save(output_dir / "input_upper.npy", upper)
    print(f"[ERA5] Pangu input ready: {output_dir}")
    return output_dir
