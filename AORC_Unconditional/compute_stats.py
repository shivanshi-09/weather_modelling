"""
compute_stats.py — run once after precompute.py, before training.
Computes mean/std for SPFH and topo over the full dataset (sampled).
Saves norm_stats.npz.
"""

import numpy as np 
import xarray as xr
from pathlib import Path

# ──────────────────────────────────────────────
AORC_ZARRS = [
    "/Datastorage/scdlds_anirudhavireddy/aorc_2020_jan_jun_3hr.zarr",
    "/Datastorage/scdlds_anirudhavireddy/aorc_2020_jul_dec_3hr.zarr",
]
TOPO_NC   = "topo_conus.nc"
OUT_STATS = "norm_stats.npz"
N_SAMPLE  = 50   # number of random timesteps to sample for stats
# ─────────────────────────────────────────────────

def welford_update(count, mean, M2, new_vals):
    "handle large arrays without storing all"
    new_vals = new_vals[~np.isnan(new_vals)].astype(np.float32)
    for x in new_vals.ravel():
        count +=1
        delta = x - mean
        mean +=delta/count
        M2 += delta *(x - mean)
    return count, mean, M2

def main():
    if Path(OUT_STATS).exists():
        print(f"Stats already exist at {OUT_STATS}, skipping")
        return 
    print("Opening AORC zarrs")
    ds = xr.concat([xr.open_zarr(p, consolidated = False) for p in AORC_ZARRS], 
        dim = "time")
    T = len (ds.time)
    rng = np.random.default_rng(42)
    t_sample = rng.choice(T, size = min (N_SAMPLE, T), replace = False)

    #____________SPFH Stats____________________________________
    print(f"Computing SPFH stats over {len(t_sample)} sampled timesteps...")

    count = 0
    sum_x = 0.0
    sum_x2 = 0.0

    for t in t_sample:
        slab = ds["SPFH_2maboveground"][int(t)].values

        valid = slab[~np.isnan(slab)].astype(np.float64)

        count += valid.size
        sum_x += valid.sum(dtype=np.float64)
        sum_x2 += np.square(valid, dtype=np.float64).sum(dtype=np.float64)

    spfh_mean = sum_x / count
    spfh_var = sum_x2 / count - spfh_mean**2
    spfh_std = np.sqrt(max(spfh_var, 0.0))

    print(f"  SPFH  mean={spfh_mean:.6f}  std={spfh_std:.6f}")

        # ── Topo stats (full grid, it's static) ───────────────────────
    print("Computing topo stats...")
    topo_ds   = xr.open_dataset(TOPO_NC)
    lat = ds.latitude.values
    lon = ds.longitude.values
    topo_fine = topo_ds["altitude"].interp(
        latitude=lat, longitude=lon, method="linear"
    ).values.astype(np.float32)

    valid     = topo_fine[~np.isnan(topo_fine)]
    topo_mean = float(valid.mean())
    topo_std  = float(valid.std())
    print(f"  Topo  mean={topo_mean:.2f}  std={topo_std:.2f}")

      # ── save ──────────────────────────────────────────────────────
    np.savez(OUT_STATS,
             spfh_mean=spfh_mean, spfh_std=spfh_std,
             topo_mean=topo_mean, topo_std=topo_std)
    print(f"Saved → {OUT_STATS}")


if __name__ == "__main__":
    main()  