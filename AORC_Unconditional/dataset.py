import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from pathlib import Path
from datetime import datetime


AORC_VARS = (
    "APCP_surface",
    "DLWRF_surface",
    "DSWRF_surface",
    "PRES_surface",
    "SPFH_2maboveground",
    "TMP_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
)


def compute_czsa(lat, lon, time):
    doy = time.timetuple().tm_yday
    hour_utc = time.hour + time.minute / 60.0
    decl = np.radians(23.45 * np.sin(np.radians(360 / 365 * (doy - 81))))
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    hour_angle = np.radians(15 * (hour_utc + lon_grid / 15.0 - 12))
    lat_r = np.radians(lat_grid)
    csza = (np.sin(lat_r) * np.sin(decl) +
            np.cos(lat_r) * np.cos(decl) * np.cos(hour_angle))
    return np.clip(csza, 0, 1).astype(np.float32)


def build_valid_index(ds, target_var, patch_size, nan_threshold=0.6, stride=128):
    var = ds[target_var]
    T, H, W = var.sizes["time"], var.sizes["latitude"], var.sizes["longitude"]
    half = patch_size // 2
    valid = []
    for t in range(T):
        slab = ds[target_var][t].values
        for i in range(half, H - half, stride):
            for j in range(half, W - half, stride):
                patch = slab[i - half:i + half, j - half:j + half]
                if np.isnan(patch).mean() < nan_threshold:
                    valid.append((t, i, j))
    return valid


class SPFHDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.zarr_paths = cfg.aorc_zarr
        self._ds = None   # lazy — opened per worker in __getitem__

        # open once just to get coords + build index
        ds = xr.concat([self._open_zarr(p) for p in self.zarr_paths], dim="time")
        self.times = ds.time.values
        self.lat   = ds.latitude.values
        self.lon   = ds.longitude.values

        # topo (static, small enough to keep in memory)
        topo_ds = xr.open_dataset(cfg.topo_nc)
        topo_coarse = topo_ds["altitude"].interp(
            latitude=ds.latitude, longitude=ds.longitude, method="linear"
        ).values.astype(np.float32)
        ds.close()

        # svf
        svf_ds = xr.open_dataset(cfg.svf_nc)
        self.svf  = svf_ds["svf"].values.astype(np.float32)
        self.topo = topo_coarse

        # valid index
        index_cache = Path(cfg.output_dir) / "valid_index.npy"
        if index_cache.exists():
            self.index = np.load(index_cache, allow_pickle=True).tolist()
        else:
            print("Building valid patch index (one time)...")
            _ds = xr.concat([self._open_zarr(p) for p in self.zarr_paths], dim="time")
            self.index = build_valid_index(_ds, cfg.target_var, self.patch_size, cfg.nan_threshold)
            index_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(index_cache, self.index)
            _ds.close()
        print(f"Total valid patches: {len(self.index)}")

        # train/val split
        all_t   = sorted(set(t for t, _, _ in self.index))
        n_val   = max(1, int(len(all_t) * cfg.val_fraction))
        t_set   = set(all_t[:-n_val]) if split == "train" else set(all_t[-n_val:])
        self.index = [(t, i, j) for t, i, j in self.index if t in t_set]

        # subsample
        cap = cfg.max_patches if split == "train" else cfg.max_val_patches
        if cap and len(self.index) > cap:
            rng  = np.random.default_rng(cfg.seed)
            idxs = rng.choice(len(self.index), size=cap, replace=False)
            self.index = [self.index[k] for k in idxs]
        print(f"  {split} patches after subsample: {len(self.index)}")

        stats = np.load(cfg.stats_file)
        self.spfh_mean = float(stats["spfh_mean"])
        self.spfh_std  = float(stats["spfh_std"])
        self.topo_mean = float(stats["topo_mean"])
        self.topo_std  = float(stats["topo_std"])

    def _open_zarr(self, path):
        drop_variables = [name for name in AORC_VARS if name != self.cfg.target_var]
        return xr.open_zarr(
            path,
            consolidated=False,
            drop_variables=drop_variables,
        )

    def _get_ds(self):
        if self._ds is None:
            """open zarr lazily"""
            print("OPENING ZARR IN WORKER")
            self._ds = xr.concat(
                [self._open_zarr(p) for p in self.zarr_paths],
                dim="time"
            )
            print("ZARR OPENED")
        return self._ds

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        t_idx, i, j = self.index[idx]
        half = self.patch_size // 2
        lat_sl = slice(i - half, i + half)
        lon_sl = slice(j - half, j + half)
        sl = (lat_sl, lon_sl)

        ds = self._get_ds()

        spfh = ds[self.cfg.target_var].isel(
            time=t_idx,
            latitude=lat_sl,
            longitude=lon_sl,
        ).values.astype(np.float32)

        spfh = (spfh - self.spfh_mean) / self.spfh_std
        spfh = np.nan_to_num(spfh, nan=0.0)

        topo = (self.topo[sl] - self.topo_mean) / self.topo_std
        svf = self.svf[sl]

        lat = self.lat[i - half:i + half]
        lon = self.lon[j - half:j + half]
        ts = self.times[t_idx].astype("datetime64[s]").astype(datetime)

        csza = compute_czsa(lat, lon, ts)

        cond = np.stack([topo, svf, csza], axis=0)

        return torch.from_numpy(np.ascontiguousarray(cond)), torch.from_numpy(np.ascontiguousarray(spfh[None]))
