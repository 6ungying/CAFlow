# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import pickle
from pathlib import Path
import torch

from guided_diffusion.script_util import create_model

from . import util
from .ckpt_util import I2SB_IMG256_UNCOND_PKL
from project_paths import DATA_DIR

from ipdb import set_trace as debug
from torch import nn

class Image256Net(torch.nn.Module):
    def __init__(self, log, noise_levels, use_fp16=False, cond=False, pretrained_adm=True, ckpt_dir=None, spm=False):
        super(Image256Net, self).__init__()

        # initialize model
        ckpt_dir = Path(ckpt_dir).expanduser().resolve() if ckpt_dir else DATA_DIR
        ckpt_pkl = ckpt_dir / I2SB_IMG256_UNCOND_PKL
        
        with open(ckpt_pkl, "rb") as f:
            kwargs = pickle.load(f)
        
        kwargs["use_fp16"] = use_fp16
        kwargs["num_channels"] = 128
        kwargs['image_size'] = 256
        kwargs['num_heads'] = 8
        kwargs["out_channels"] = 1
        kwargs["in_channels"] = 1
        
        self.cond = cond
        self.diffusion_model = create_model(**kwargs)
        
        # Keep the denoising backbone depth-only.
        first_conv = self.diffusion_model.input_blocks[0][0]
        if first_conv.in_channels != 1:
            new_conv = torch.nn.Conv2d(
                in_channels=1,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            
            with torch.no_grad():
                old_weight = first_conv.weight
                new_conv.weight.copy_(old_weight.mean(dim=1, keepdim=True))
                
                if first_conv.bias is not None:
                    new_conv.bias.copy_(first_conv.bias)
            
            self.diffusion_model.input_blocks[0][0] = new_conv
            log.info("[Net] Modified first layer to accept one depth channel")
        
        log.info(f"[Net] Initialized depth-only network from {ckpt_pkl=}! Size={util.count_parameters(self.diffusion_model)}!")

        self.diffusion_model.apply(self.init_weights)
        self.diffusion_model.eval()
        
        self.spm = spm

        if self.cond:
            # xt depth (1) + DEM (1) -> depth feature (1)
            self.cond_adapter = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
            nn.init.kaiming_normal_(self.cond_adapter.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(self.cond_adapter.bias)
            log.info("[Net] Added DEM conditioning adapter (2->1 channels)")

        if self.spm:
            self.spm_adapter = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
            nn.init.kaiming_normal_(self.spm_adapter.weight, mode='fan_out', nonlinearity='relu')
            if self.spm_adapter.bias is not None:
                nn.init.zeros_(self.spm_adapter.bias)
            log.info("[Net] Added SPM adapter (2->1 channels)")
        
        self.noise_levels = noise_levels

    def forward(self, x, steps, rainfall, cond=None, spm=None):

        t = steps.detach()
        if t.dim() > 1:
            t = t.view(t.shape[0], -1)[:, 0]
        t = t.to(dtype=torch.long, device=x.device)
        assert t.dim() == 1 and t.shape[0] == x.shape[0]

        if self.cond:
            if cond is None:
                raise ValueError("DEM conditioning is enabled, but cond is None")
            if not isinstance(cond, torch.Tensor):
                cond = torch.as_tensor(cond, device=x.device)
            else:
                cond = cond.to(device=x.device)
            if cond.dim() == 2:
                cond = cond.unsqueeze(0).unsqueeze(0)
            elif cond.dim() == 3:
                cond = cond.unsqueeze(1) if cond.shape[0] == x.shape[0] else cond.unsqueeze(0)
            if cond.dim() != 4:
                raise ValueError(f"DEM condition must be 4D after batching, got shape={tuple(cond.shape)}")
            cond = cond[:, 0:1, ...]
            if cond.shape[0] == 1 and x.shape[0] > 1:
                cond = cond.expand(x.shape[0], -1, -1, -1)
            if cond.shape[0] != x.shape[0] or cond.shape[2:] != x.shape[2:]:
                raise ValueError(f"DEM condition shape {tuple(cond.shape)} does not match x shape {tuple(x.shape)}")
            cond = cond.to(dtype=x.dtype)
            x = self.cond_adapter(torch.cat([x, cond], dim=1))

        if self.spm and spm is not None:
            # SPM integration (保留原始邏輯)
            if not isinstance(spm, torch.Tensor):
                spm = torch.as_tensor(spm, device=x.device)
            else:
                spm = spm.to(device=x.device)

            # 維度處理 (SPM 是單通道)
            if spm.dim() == 2:
                spm = spm.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
            elif spm.dim() == 3:
                if spm.shape[0] == x.shape[0]:
                    spm = spm.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
                else:
                    spm = spm.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
            elif spm.dim() == 4:
                if spm.shape[1] != 1:
                    spm = spm[:, 0:1, ...]  # Take first channel

            # Broadcast batch dim
            if spm.shape[0] == 1 and x.shape[0] > 1:
                spm = spm.expand(x.shape[0], -1, -1, -1)

            # Ensure dtype matches
            if spm.dtype != x.dtype:
                spm = spm.to(dtype=x.dtype)

            x = torch.cat([x, spm], dim=1)
            x = self.spm_adapter(x)

        output = self.diffusion_model(x, t, rainfall)
        return output  # [batch, 1, H, W]

    
    def init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
