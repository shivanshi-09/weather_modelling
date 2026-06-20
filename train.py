"""
train.py — EDM diffusion training loop for SPFH.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import time
from gpu_monitor import GPUMonitor

from config import CFG
from dataset import TargetDataset
from model import build_model
from diffusion import EDM


def train():
    torch.manual_seed(CFG.seed)
    device = torch.device(CFG.device)
    out_dir = Path(CFG.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ── data ──────────────────────────────────────────────────────
    print("Building datasets...")
    train_ds = TargetDataset(CFG, split="train")
    val_ds   = TargetDataset(CFG, split="val")

    loader_kwargs = {
        "num_workers": CFG.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": CFG.num_workers > 0,
    }
    if CFG.num_workers > 0:
        loader_kwargs["prefetch_factor"] = CFG.prefetch_factor

    train_dl = DataLoader(
        train_ds, batch_size=CFG.batch_size,
        shuffle=True, **loader_kwargs,
    )
    val_dl = DataLoader(
        val_ds, batch_size=CFG.batch_size,
        shuffle=False, **loader_kwargs,
    )
    print(f"Train patches: {len(train_ds)}  Val patches: {len(val_ds)}")

    # ── model + diffusion ─────────────────────────────────────────
    print("Building model...", flush=True)
    model = build_model().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        if CFG.compile_model and device.type == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("Model compiled with torch.compile (reduce-overhead)", flush=True)
            except Exception as exc:
                print(f"torch.compile unavailable ({exc}), continuing without.", flush=True)

    print("Model built", flush=True)
    gpu_id = device.index if device.index is not None else 0
    monitor = GPUMonitor(gpu_id=gpu_id, interval=1.0, window=10)
    print(f"GPU monitor started (target util {CFG.gpu_util_low}-{CFG.gpu_util_high}%)", flush=True)

           
    edm   = EDM(sigma_data=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    amp_dtype = torch.bfloat16 if CFG.amp_dtype == "bf16" else torch.float16
    use_scaler = CFG.amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ── resume if checkpoint exists ───────────────────────────────
    start_epoch = 0
    ckpt_path   = out_dir / "last.pt"
    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
        except RuntimeError as exc:
            print(
                f"Checkpoint is incompatible with current model config; "
                f"starting from scratch. Details: {exc}",
                flush=True,
            )
    # ── training ──────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, CFG.num_epochs):
            print(f"Epoch {epoch} start", flush=True)
            model.train()
            epoch_loss = 0.0
            t0 = time.time()
            last_log = t0
            optimizer.zero_grad(set_to_none=True)
            accum = CFG.grad_accum_steps
            for step, (cond, target) in enumerate(train_dl):
                cond   = cond.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                target = target.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=CFG.amp):
                    loss = edm.loss(model, target, cond) / accum

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % accum == 0 or (step + 1) == len(train_dl):
                    if use_scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                step_loss = loss.item() * accum
                epoch_loss += step_loss
                if step % CFG.log_every == 0:
                    now = time.time()
                    secs = now - last_log
                    samples_per_sec = (CFG.log_every * CFG.batch_size / secs) if step else 0.0
                    gpu_str = monitor.log_and_warn(step, CFG.gpu_util_low, CFG.gpu_util_high)
                    print(
                        f"epoch {epoch} step {step}/{len(train_dl)} "
                        f"loss {loss.item():.6f} "
                        f"dt {secs:.1f}s samples/s {samples_per_sec:.2f} | {gpu_str}",
                        flush=True,
                    )
                    last_log = now

            # ── val ───────────────────────────────────────────────────
            do_val = (epoch == start_epoch) or ((epoch + 1) % CFG.val_every == 0) or (epoch + 1 == CFG.num_epochs)
            val_loss = None
            if do_val:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for cond, target in val_dl:
                        cond = cond.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                        target = target.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=CFG.amp):
                            val_loss += edm.loss(model, target, cond).item()
                val_loss = val_loss / len(val_dl)

            train_loss = epoch_loss / len(train_dl)
            elapsed    = time.time() - t0
            val_text = f"{val_loss:.6f}" if val_loss is not None else "skipped"
            print(f"Epoch {epoch:03d} | train {train_loss:.6f} | "
                f"val {val_text} | {elapsed:.1f}s")

            # ── checkpoint ────────────────────────────────────────────
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, ckpt_path)
            if (epoch + 1) % CFG.save_every == 0:
                torch.save({"epoch": epoch, "model": model.state_dict()},
                        out_dir / f"ckpt_epoch{epoch:03d}.pt")
    finally: 
        monitor.stop()


if __name__ == "__main__":
    train()
