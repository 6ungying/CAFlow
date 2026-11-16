"""Pixel-weighted water-depth metrics before PNG quantization."""

import math
import torch


def depth_metres(normalized, mean, std, max_depth):
    pixels = normalized * std + mean
    scale = torch.as_tensor(max_depth, device=pixels.device, dtype=pixels.dtype)
    if scale.ndim == 1:
        scale = scale.view(-1, 1, 1, 1)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Maximum depth must be finite and positive.")
    # Dataset encoding: white = dry, black = maximum water depth.
    return (1.0 - pixels) * scale


class DepthMetrics:
    def __init__(self):
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.pixels = 0

    def update(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes must match.")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("Non-finite depth encountered during validation.")
        error = (prediction - target).double()
        self.absolute_error += error.abs().sum().item()
        self.squared_error += error.square().sum().item()
        self.pixels += error.numel()

    def compute(self):
        if not self.pixels:
            raise ValueError("Validation dataset is empty.")
        return {"mae_m": self.absolute_error / self.pixels,
                "rmse_m": math.sqrt(self.squared_error / self.pixels)}
