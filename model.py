"""
model.py- thin wrapper for PhysicsNeMo's diffusion UNet
"""

"""
Input 4 x 256 x 256

↓
64 x 256 x 256

↓
128 x 128 x 128

↓
256 x 64 x 64

↓
512 x 32 x 32 (attention)

↓
512 x 16 x 16 (attention)

↑
512 x 32 x 32 (attention)

↑
256 x 64 x 64

↑
128 x 128 x 128

↑
64 x 256 x 256

↓
1 x 256 x 256
"""
"""
model.py - thin wrapper for PhysicsNeMo's SongUNet (true EDM diffusion,
noise-level conditioning via noise_labels, not the CorrDiff regression UNet).

SongUNet.forward signature (confirmed on installed version):
    forward(self, x, noise_labels, class_labels, augment_labels=None)

There is NO separate `cond` argument -- conditioning channels (topo, SVF,
CSZA) must be concatenated onto the noisy input yourself, before calling
the model. in_channels in __init__ is the TOTAL channel count of that
concatenated tensor, not just the noisy portion.

So: in_channels = 1 (noisy temp) + 3 (cond channels) = 4
    out_channels = 1 (denoised temp)
"""
import torch
import torch.nn
from physicsnemo.models.diffusion.song_unet import SongUNet
from config import CFG


def build_model():
    model = SongUNet(
        img_resolution=CFG.patch_size,
        in_channels=5,            # 1 noisy x + 3 conditioning channels, concatenated by you before forward()
        out_channels=1,
        model_channels=CFG.base_ch,
        channel_mult=list(CFG.ch_mult),
        num_blocks=CFG.num_res_blocks,
        attn_resolutions=list(CFG.attn_resolutions),
        label_dim=0,               # no class conditioning
        embedding_type="positional",
        amp_mode=CFG.amp,
    )
    return model


if __name__ == "__main__":
    model = build_model()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"parameters {total:.1f}M")

    x_noisy = torch.randn(2, 1, 256, 256)
    cond = torch.randn(2, 3, 256, 256)

    # YOU concatenate -- the model never sees x_noisy and cond separately
    x_input = torch.cat([x_noisy, cond], dim=1)  # (2, 4, 256, 256)

    # noise_labels is EDM's c_noise(sigma), not raw sigma or a timestep index.
    # Quick smoke-test stand-in here -- wire up the real EDM preconditioning
    # (c_skip/c_out/c_in/c_noise from sigma) in the training loop, not here.
    sigma = torch.rand(2) * 10  # placeholder
    noise_labels = sigma.log() / 4  # placeholder EDM-style noise embedding input, replace with real c_noise(sigma)
    class_labels = None  # label_dim=0, so this can stay None

    out = model(x_input, noise_labels, class_labels)
    print(f"Input: {x_input.shape} -> Output: {out.shape}")
