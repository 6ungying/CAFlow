# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
from tqdm import tqdm

import distributed_util as dist_util
from matplotlib import pyplot as plt

from . import util
from .network import Image256Net
from .diffusion import Diffusion, disabled_train, create_model_config
from i2sb.VQGAN.vqgan import VQModel
from i2sb.base.modules.encoders.modules import SpatialRescaler
from torch.utils.data import DataLoader
from corruption.mixture import floodDataset
from .embedding import RainfallEmbedder
from .training_plot import save_training_error_plot
from .depth_metrics import DepthMetrics, depth_metres
from .training_contract import OBJECTIVE, training_contract, check_checkpoint_contract

from ipdb import set_trace as debug

def build_optimizer_sched(opt, rainfall_embber, net, log):
    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    params = list(net.parameters()) + list(rainfall_embber.parameters())
    optimizer = AdamW(params, **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])

    return optimizer, sched

def normalize_and_clip_gradients(parameters, valid_samples, max_norm):
    if valid_samples <= 0:
        raise ValueError("valid_samples must be positive.")
    parameters = list(parameters)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(valid_samples)
    # The joint norm check rejects NaN/Inf before clipping, with one GPU sync.
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)


def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def plot_grad_flow(model):
    ave_grads = []
    layers = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    fig = plt.figure(figsize=(10,5))
    plt.plot(ave_grads, alpha=0.5, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    # plt.xticks(range(0,len(ave_grads),1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient Magnitude")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.show()

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        if getattr(opt, 'timestep_importance', None) != 'discrete':
            raise ValueError('Only discrete I2SB residual checkpoints are supported; retrain legacy continuous runs.')
        if opt.clip_denoise or opt.latent_space or opt.add_x1_noise:
            raise ValueError('Depth I2SB requires pixel space without [-1,1] clipping or extra x1 noise.')
        opt.objective = OBJECTIVE
        checkpoint = torch.load(opt.load, map_location='cpu') if opt.load else None
        if checkpoint is not None:
            check_checkpoint_contract(checkpoint, opt)
        self.start_step = checkpoint.get('step', 0) if checkpoint is not None else 0
        self.best_rmse = checkpoint.get('best_rmse_m', float('inf')) if checkpoint is not None else float('inf')

        self.opt = opt
        self.model_config = create_model_config()
        if opt.latent_space:
            self.vqgan = VQModel(**vars(self.model_config.VQGAN.params)).eval()
            self.vqgan.train = disabled_train
            for param in self.vqgan.parameters():
                param.requires_grad = False
            self.cond_stage_model = SpatialRescaler(**vars(self.model_config.CondStageParams))
            self.vqgan.to(opt.device)
            self.cond_stage_model.to(opt.device)

        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=True)
        self.rainfall_emb = RainfallEmbedder(256, 1)
        params = list(self.net.parameters()) + list(self.rainfall_emb.parameters())
        self.ema = ExponentialMovingAverage(params, decay=opt.ema)

        if opt.load:
            net = checkpoint['net']
            for key in net.keys():
                if 'cond_adapter' in key:
                    log.info(f"[Net] Detected DEM conditioning checkpoint key: {key}")

            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            self.rainfall_emb.load_state_dict(checkpoint['embedding'])
            if opt.normalize_latent:
                self.net.ori_latent_mean = checkpoint["ori_latent_mean"]
                self.net.ori_latent_std = checkpoint["ori_latent_std"]
                self.net.cond_latent_mean = checkpoint["cond_latent_mean"]
                self.net.cond_latent_std = checkpoint["cond_latent_std"]

        self.net.to(opt.device)
        self.ema.to(opt.device)
        self.rainfall_emb.to(opt.device)

        self.log = log

        if opt.eval and opt.normalize_latent:
            self.net.ori_latent_mean = self.net.ori_latent_mean.to(opt.device)
            self.net.ori_latent_std = self.net.ori_latent_std.to(opt.device)
            self.net.cond_latent_mean = self.net.cond_latent_mean.to(opt.device)
            self.net.cond_latent_std = self.net.cond_latent_std.to(opt.device)

    def logger(self, msg, **kwargs):
        print(msg, **kwargs)

    def get_latent_mean_std(self): pass
    def encode(self, x, cond=True): pass
    def decode(self, x_latent, cond=True): pass

    def compute_label(self, step, x0, xt, x1):
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, x1, net_out, clip_denoise=False):
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise:
            raise ValueError('Depth z-scores are not bounded by [-1,1]; disable clip_denoise.')
        return pred_x0

    def sample_batch(self, opt, loader, corrupt_method):
        (
            flood_img, dem_img, mask_flood, y, img_name,
            ca4d, max_depth, dem_id
        ) = next(loader)

        def ensure_single_channel(t):
            if t.dim() == 4 and t.shape[1] > 1:
                return t[:, 0:1, :, :]
            return t

        flood_img = ensure_single_channel(flood_img)
        dem_img = ensure_single_channel(dem_img)

        if not getattr(opt, "ca4d", False):
            raise ValueError("Depth-only PIFF requires CA4D as x1; enable --ca4d")
        if ca4d is None:
            raise ValueError("Depth-only PIFF requires a paired CA4D depth image for every sample")

        clean_img = flood_img
        corrupt_img = ensure_single_channel(ca4d)

        # [MODIFIED] 強制設定 Mask 為 None (完全不使用 Mask)
        # 即使 dataset 回傳了 None，我們也明確指定 mask 變數為 None，
        # 避免後續程式碼誤用 mask 進行運算
        mask = None

        device = opt.device
        y      = y.to(device)
        x0     = clean_img.to(device)
        x1     = corrupt_img.to(device)
        cond = dem_img.to(device)

        if getattr(opt, "add_x1_noise", False):
            x1 = x1 + torch.randn_like(x1)

        return x0, x1, mask, y, cond, max_depth, dem_id

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        if opt.global_size != 1:
            raise ValueError("Distributed gradient synchronization is not implemented.")
        self.writer = util.build_log_writer(opt)
        optimizer, sched = build_optimizer_sched(opt, self.rainfall_emb, self.net, self.log)
        train_loader = util.setup_loader(train_dataset, opt.microbatch, opt.num_workers)
        if len(train_dataset) < opt.microbatch:
            raise ValueError("Training dataset is smaller than microbatch.")
        val_loader = DataLoader(
            val_dataset, batch_size=opt.microbatch, shuffle=False,
            pin_memory=True, num_workers=opt.num_workers, drop_last=False,
            collate_fn=util.custom_collate_fn,
        )
        parameters = list(self.net.parameters()) + list(self.rainfall_emb.parameters())
        inner_steps = opt.batch_size // opt.microbatch
        loss_steps, losses = [], []
        loss_plot_path = opt.log_dir / opt.name / "training_error.png"
        step = self.start_step

        def save_checkpoint(filename):
            torch.save({
                "net": self.net.state_dict(), "embedding": self.rainfall_emb.state_dict(),
                "ema": self.ema.state_dict(), "optimizer": optimizer.state_dict(),
                "sched": sched.state_dict() if sched is not None else None,
                "step": step, "best_rmse_m": self.best_rmse,
                "training_contract": training_contract(opt),
            }, opt.ckpt_path / filename)

        try:
            self.net.train()
            self.rainfall_emb.train()
            for attempt in range(self.start_step, opt.num_itr):
                optimizer.zero_grad(set_to_none=True)
                weighted_loss, valid_samples = 0.0, 0
                for _ in range(inner_steps):
                    x0, x1, _, y, cond, _, _ = self.sample_batch(opt, train_loader, corrupt_method)
                    if not all(torch.isfinite(t).all() for t in (x0, x1, cond, y)):
                        self.log.warning('Skipped non-finite input microbatch.')
                        continue
                    indices = torch.randint(opt.interval, (x0.shape[0],), device=x0.device)
                    xt = self.diffusion.q_sample(indices, x0, x1, ot_ode=opt.ot_ode)
                    label = self.compute_label(indices, x0, xt, x1)
                    pred = self.net(xt, indices, self.rainfall_emb(y), cond=cond)
                    loss = F.mse_loss(pred, label)
                    if not torch.isfinite(loss):
                        self.log.warning('Skipped non-finite loss microbatch.')
                        continue
                    count = x0.shape[0]
                    # Accumulate sample sums, then divide by the actual valid count.
                    (loss * count).backward()
                    weighted_loss += loss.detach().item() * count
                    valid_samples += count
                if not valid_samples:
                    self.log.warning('Skipped optimizer update: no valid samples.')
                    continue
                try:
                    grad_norm = normalize_and_clip_gradients(parameters, valid_samples, opt.grad_clip)
                except (FloatingPointError, RuntimeError):
                    optimizer.zero_grad(set_to_none=True)
                    self.log.warning('Skipped optimizer update: non-finite gradients.')
                    continue
                optimizer.step()
                self.ema.update()
                if sched is not None:
                    sched.step()
                step += 1
                mean_loss = weighted_loss / valid_samples
                loss_steps.append(step)
                losses.append(mean_loss)
                self.log.info(f'train_it {step}/{opt.num_itr} | lr:{optimizer.param_groups[0]["lr"]:.2e} '
                              f'| loss:{mean_loss:.6f} | valid_samples:{valid_samples} | grad_norm:{float(grad_norm):.4f}')
                self.writer.add_scalar(step, 'train/loss', mean_loss)
                self.writer.add_scalar(step, 'train/valid_samples', valid_samples)
                if step % opt.val_every == 0:
                    metrics = self.evaluation(opt, step, val_loader, corrupt_method)
                    if metrics['rmse_m'] < self.best_rmse:
                        self.best_rmse = metrics['rmse_m']
                        save_checkpoint('best.pt')
                    save_checkpoint('latest.pt')
                elif step % 100 == 0:
                    save_checkpoint('latest.pt')
                if step % 100 == 0:
                    save_training_error_plot(loss_steps, losses, loss_plot_path)
            # Include the final optimizer state even when it is off the save cadence.
            if step > self.start_step and step % opt.val_every != 0:
                metrics = self.evaluation(opt, step, val_loader, corrupt_method)
                if metrics['rmse_m'] < self.best_rmse:
                    self.best_rmse = metrics['rmse_m']
                    save_checkpoint('best.pt')
            if step > self.start_step:
                save_checkpoint('latest.pt')
                save_training_error_plot(loss_steps, losses, loss_plot_path)
        finally:
            self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, y, mask=None, cond=None, clip_denoise=False,
                      nfe=None, log_count=10, verbose=True, eval=False, ode_method=None):
        if ode_method not in (None, 'i2sb'):
            raise ValueError('Residual I2SB checkpoints require the i2sb posterior sampler.')
        if clip_denoise:
            raise ValueError('Depth z-scores cannot use [-1,1] denoising clipping.')
        nfe = opt.interval - 1 if nfe is None else nfe
        if not 1 <= nfe < opt.interval:
            raise ValueError('Require 1 <= nfe < interval.')
        steps = util.space_indices(opt.interval, nfe + 1)
        log_count = max(1, min(nfe, log_count))
        log_steps = [steps[i] for i in util.space_indices(len(steps) - 1, log_count)]
        net_training, rain_training = self.net.training, self.rainfall_emb.training
        x1 = x1.to(opt.device)
        cond = cond.to(opt.device) if cond is not None else None
        try:
            self.net.eval()
            self.rainfall_emb.eval()
            with self.ema.average_parameters():
                rainfall_emb = self.rainfall_emb(y.to(opt.device))

                def pred_x0_fn(x1, xt, rainfall_emb, index):
                    indices = torch.full((xt.shape[0],), index, device=xt.device, dtype=torch.long)
                    residual = self.net(xt, indices, rainfall_emb, cond=cond)
                    return self.compute_pred_x0(indices, xt, x1, residual)

                return self.diffusion.ddpm_sampling(
                    steps, pred_x0_fn, x1, rainfall_emb, mask=mask, ot_ode=opt.ot_ode,
                    log_steps=log_steps, verbose=verbose,
                )
        finally:
            self.net.train(net_training)
            self.rainfall_emb.train(rain_training)

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method):
        prediction_metrics, baseline_metrics = DepthMetrics(), DepthMetrics()
        dataset = val_loader.dataset
        if not hasattr(dataset, 'depth_mean') or not hasattr(dataset, 'depth_std'):
            raise ValueError('Validation dataset must declare its depth normalization.')
        device = torch.device(opt.device)
        devices = [torch.cuda.current_device() if device.index is None else device.index] if device.type == 'cuda' else []
        # Same validation noise every time; validation must not advance training RNG.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(opt.validation_seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed(opt.validation_seed)
            for batch in val_loader:
                x0, x1, _, y, cond, maximum, _ = self.sample_batch(opt, iter([batch]), corrupt_method)
                xs, _ = self.ddpm_sampling(
                    opt, x1, y, cond=cond, nfe=opt.val_nfe, log_count=1, verbose=False,
                )
                prediction = xs[:, 0].to(x0.device)
                truth_m = depth_metres(x0, dataset.depth_mean, dataset.depth_std, maximum)
                prediction_m = depth_metres(prediction, dataset.depth_mean, dataset.depth_std, maximum)
                baseline_m = depth_metres(x1, dataset.depth_mean, dataset.depth_std, maximum)
                prediction_metrics.update(prediction_m, truth_m)
                baseline_metrics.update(baseline_m, truth_m)
        metrics = prediction_metrics.compute()
        metrics.update({'ca4d_' + key: value for key, value in baseline_metrics.compute().items()})
        for key, value in metrics.items():
            self.writer.add_scalar(it, 'validation/' + key, value)
        self.log.info(f'Validation step={it} samples={len(dataset)} nfe={opt.val_nfe}: {metrics}')
        return metrics
