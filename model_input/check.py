import numpy as np
import xarray as xr
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
case_dir = PROJECT_ROOT / "model_input" / "single_time_point" / "era5" / "2025-07-01-00-00"
nc_path = case_dir / "surface.nc"
npy_path = case_dir / "input_surface.npy"

ds = xr.open_dataset(nc_path)
arr = np.load(npy_path)

t2m_nc = ds["t2m"].isel(valid_time=0).values
t2m_npy = arr[3]

print("npy shape:", arr.shape)
print("nc t2m shape:", t2m_nc.shape)
print("max abs diff:", np.nanmax(np.abs(t2m_npy - t2m_nc)))
print("mean abs diff:", np.nanmean(np.abs(t2m_npy - t2m_nc)))

ds.close()
