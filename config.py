from dataclasses import dataclass, field

@dataclass

class Config: 
    aorc_zarr: list = field(default_factory=lambda: [
    "/Datastorage/scdlds_anirudhavireddy/aorc_2020_jan_jun_3hr.zarr",
    "/Datastorage/scdlds_anirudhavireddy/aorc_2020_jul_dec_3hr.zarr", 
])
    topo_nc: str = "topo_conus.nc"
    svf_nc: str = "svf_conus.nc"
    stats_file: str = "norm_stats.npz"
    output_dir: str = "runs/spfh_edm"
    era5_dir: str = "/Datastorage/shivanshi.singh_ug2024/Diffusion/era5_conus_temp_arco"
    era5_var: str = "t2m"

    target_var: str = "TMP_2maboveground"
    patch_size: int = 256
    val_fraction: float = 0.1
    nan_threshold = 0.6

    # --------norm stats (from compute_stats.py)
    spfh_mean: float = 0.006544
    spfh_std:  float = 0.004416
    topo_mean: float = -833.606
    topo_std:  float = 2315.990

    # --- Diffusion ---
    T:          int   = 1000
    beta_start: float = 1e-4
    beta_end:   float = 0.02
    schedule:   str   = "cosine" 

    # --- Model ---
    base_ch:            int   = 64
    ch_mult:            tuple = (1, 2, 4, 8)
    num_res_blocks:     int   = 2
    attn_resolutions:   tuple = (32, 16)

    # --- Training ---
    max_patches: int = 25000
    max_val_patches: int = 2000
    batch_size:  int = 16
    lr:          float = 1e-4
    num_epochs:  int   = 20
    num_workers: int   = 4
    prefetch_factor: int = 4
    grad_clip:   float = 1.0
    save_every:  int   = 5
    val_every:   int   = 5
    log_every:   int   = 10
    seed:        int   = 42
    amp:         bool  = True
    amp_dtype:   str   = "bf16"
    device:      str   = "cuda"
    compile_model: bool = True
    grad_accum_steps: int = 1

    # --- GPU monitoring ---
    gpu_util_low:  int = 90
    gpu_util_high: int = 95

CFG = Config()
