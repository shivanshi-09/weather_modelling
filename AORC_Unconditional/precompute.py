"""
precompute.py
Saves:
  - svf_conus.nc : (H, W) float32
"""

import numpy as np
import xarray as xr
from pathlib import Path


# ── paths ────────────────────────────────────────────
AORC_ZARRS = [
    "/Datastorage/scdlds_anirudhavireddy/aorc_2020_jan_jun_3hr.zarr",
    "/Datastorage/scdlds_anirudhavireddy/aorc_2020_jul_dec_3hr.zarr",
]
TOPO_NC  = "topo_conus.nc"
OUT_SVF  = "svf_conus.nc"
# ──────────────────────────────────────────────────────────────────


def compute_svf(altitude, res_m=927.0):
    """
    Slope-based SVF approximation: SVF ≈ cos(slope).
    altitude : (H, W) float32 in metres
    res_m    : pixel size in metres
    """
    dz_dy = np.gradient(altitude, res_m, axis=0)
    dz_dx = np.gradient(altitude, res_m, axis=1)
    slope  = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    return np.cos(slope).astype(np.float32)


def main():
    if Path(OUT_SVF).exists():
        print(f"SVF already exists at {OUT_SVF}, skipping.")
        return

    print("Loading AORC coords for target grid...")
    ds  = xr.open_zarr(AORC_ZARRS[0], consolidated=False)
    lat = ds.latitude.values
    lon = ds.longitude.values

    print("Loading and upsampling topo...")
    topo_ds   = xr.open_dataset(TOPO_NC)
    topo_fine = topo_ds["altitude"].interp(
        latitude=lat,
        longitude=lon,
        method="linear"
    ).values.astype(np.float32)              # (4201, 8401)

    print("Computing SVF...")
    svf = compute_svf(topo_fine)

    print("Saving SVF...")
    xr.DataArray(
        svf,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
        name="svf"
    ).to_netcdf(OUT_SVF)

    print(f"Done — saved SVF → {OUT_SVF}")


if __name__ == "__main__":
    main()