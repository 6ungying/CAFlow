# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import random
import argparse

import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.multiprocessing import Process

from logger import Logger
from distributed_util import init_processes
from corruption import build_corruption
from corruption.mixture import floodDataset, singleDEMFloodDataset
from dataset import imagenet
from i2sb import Runner, download_ckpt
from project_paths import DATA_DIR, LOG_DIR, RESULTS_DIR

import colored_traceback.always
from ipdb import set_trace as debug

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--name",           type=str,   default=None, help="experiment ID; defaults to a new I2SB run name")
    parser.add_argument("--ckpt",           type=str,   default=None,        help="resumed checkpoint name")
    parser.add_argument("--gpu",            type=int,   default=None,        help="set only if you wish to run on a particular device; use -1 for CPU")
    parser.add_argument("--n-gpu-per-node", type=int,   default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,   default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,   default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,   default=1,           help="The number of nodes in multi node env")
    parser.add_argument("--eval",           action="store_true", default=False,         help="evaluation mode")
    # parser.add_argument("--amp",            action="store_true")

    # --------------- SB model ---------------
    parser.add_argument("--image-size",     type=int,   default=256)
    parser.add_argument("--dataset-dir", type=Path, default=DATA_DIR / "dems", help="path to dataset")
    parser.add_argument("--latent-space", action="store_true", default=False, help="use latent space model")
    parser.add_argument("--normalize-latent", action="store_true", default=False, help="normalize latent space")
    parser.add_argument("--timestep-importance", choices=['discrete'], default='discrete', help="I2SB discrete residual objective")
    parser.add_argument("--corrupt",        type=str,   default='mixture',        help="restoration task")
    parser.add_argument("--t0",             type=float, default=1e-4,        help="sigma start time in network parametrization")
    parser.add_argument("--T",              type=float, default=1.,          help="sigma end time in network parametrization")
    parser.add_argument("--interval",       type=int,   default=1000,        help="number of interval")
    parser.add_argument("--beta-max",       type=float, default=0.1,         help="max diffusion for the diffusion model")
    # parser.add_argument("--beta-min",       type=float, default=0.1)
    parser.add_argument("--ot-ode",         action="store_true",  default=False,           help="use OT-ODE model")
    parser.add_argument("--clip-denoise",   action="store_true",             help="clamp predicted image to [-1,1] at each")

    # CA4D is x1; DEM conditioning is always enabled.
    parser.add_argument("--ca4d",           action="store_true",  default=True,           help="use paired CA4D depth as I2SB x1")
    
    parser.add_argument("--add-x1-noise",   action="store_true",             help="add noise to conditional network")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--microbatch",     type=int,   default=4,           help="accumulate gradient over microbatch until full batch-size")
    parser.add_argument("--num-itr",        type=int,   default=1000000,     help="training iteration")
    parser.add_argument("--lr",             type=float, default=5e-5,        help="learning rate")
    parser.add_argument("--lr-gamma",       type=float, default=0.99,        help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,        help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0)
    parser.add_argument("--ema",            type=float, default=0.999)

    # --------------- path and logging ---------------
    parser.add_argument("--log-dir",        type=Path,  default=LOG_DIR,      help="path to log std outputs and writer data")
    parser.add_argument("--log-writer", choices=['tensorboard', 'wandb', 'none'], default='tensorboard')
    parser.add_argument("--wandb-api-key",  type=str,   default=None,        help="unique API key of your W&B account; see https://wandb.ai/authorize")
    parser.add_argument("--wandb-user",     type=str,   default=None,        help="user name of your W&B account")
    
    # --------------- dataset selection ---------------
    parser.add_argument("--use-single-dem", action="store_true", default=False, help="use single DEM flood dataset instead of multi-DEM")
    
    # --------------- test DEM configuration ---------------
    parser.add_argument("--test-dem-list",  type=str,   default=None,        help="Comma-separated list of test DEM numbers to exclude from training, e.g., '8,29,62'")
    parser.add_argument("--test-rain-list", type=str,   default=None,        help="Comma-separated list of test rainfall scenarios to exclude from training, e.g., '9,20,33'")
    parser.add_argument('--val-rain-list', type=str, default=None, help='Held-out training rainfall event IDs, comma-separated')
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--validation-seed', type=int, default=12345)
    parser.add_argument('--val-every', type=int, default=1000, help='Validate the complete holdout every N optimizer steps')
    parser.add_argument('--val-nfe', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--single-max-depth', type=float, default=4.0, help='Single DEM PNG maximum depth in metres; white is dry')
    parser.add_argument('--grad-clip', type=float, default=0.5)

    opt = parser.parse_args()
    if opt.n_gpu_per_node != 1 or opt.num_proc_node != 1:
        parser.error('This training loop currently supports one GPU only.')
    if opt.interval < 2 or opt.interval % 2:
        parser.error('--interval must be an even integer >= 2.')
    if opt.image_size != 256 or opt.beta_max <= 0:
        parser.error('This backbone requires image-size=256 and a positive beta-max.')
    if not (1 <= opt.val_nfe < opt.interval) or opt.val_every < 1:
        parser.error('Require 1 <= val-nfe < interval and val-every >= 1.')
    if opt.microbatch < 1 or opt.batch_size < 1 or opt.batch_size % opt.microbatch:
        parser.error('Batch size must be positive and divisible by microbatch.')
    if opt.num_workers < 0 or opt.num_itr < 1 or opt.single_max_depth <= 0 or opt.grad_clip <= 0:
        parser.error('Invalid workers, iterations, depth scale or gradient clipping value.')
    if opt.latent_space or opt.normalize_latent or opt.clip_denoise or opt.add_x1_noise or opt.eval:
        parser.error('The corrected depth I2SB path uses pixel space without [-1,1] clipping, x1 noise or --eval.')
    if not 0 < opt.val_fraction < 1:
        parser.error('--val-fraction must be between zero and one.')
    opt.val_rain_list = ([int(x.strip()) for x in opt.val_rain_list.split(',')]
                         if opt.val_rain_list else None)
    
    # ========= process test DEM list =========
    if opt.test_dem_list:
        opt.test_dem_list = [int(x.strip()) for x in opt.test_dem_list.split(',')]
        print(f"Test DEMs (excluded from training): {opt.test_dem_list}")
    else:
        opt.test_dem_list = None
    
    # ========= process test rainfall list =========
    if opt.test_rain_list:
        opt.test_rain_list = [int(x.strip()) for x in opt.test_rain_list.split(',')]
        print(f"Test Rainfall Scenarios (excluded from training): {opt.test_rain_list}")
    else:
        opt.test_rain_list = None

    # ========= auto setup =========
    # 嘗試使用 GPU（即使有 sm_120 警告也試試看）
    if opt.gpu is None:
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        opt.device = f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print(f"Using device: {opt.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("Note: sm_120 warning can be ignored - will try to run anyway")
    print("=" * 60)
    
    if opt.name is None:
        opt.name = opt.ckpt or ('flood-single-i2sb' if opt.use_single_dem else 'flood-multi-i2sb')
    opt.distributed = opt.n_gpu_per_node > 1
    opt.use_fp16 = False # disable fp16 for training

    # log ngc meta data
    if "NGC_JOB_ID" in os.environ.keys():
        opt.ngc_job_id = os.environ["NGC_JOB_ID"]

    # ========= path handle =========
    opt.ckpt_path = RESULTS_DIR / opt.name
    if opt.ckpt_path.exists() and any(opt.ckpt_path.iterdir()) and opt.ckpt is None:
        parser.error('Experiment directory already contains files; use a new --name or explicit --ckpt.')

    if opt.ckpt is not None:
        ckpt_file = RESULTS_DIR / opt.ckpt / "latest.pt"
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None

    # ========= auto assert =========
    assert opt.batch_size % opt.microbatch == 0, f"{opt.batch_size=} is not dividable by {opt.microbatch}!"
    return opt

def main(opt):
    run_log_dir = opt.log_dir / opt.name
    log = Logger(opt.global_rank, run_log_dir, append=True)
    log.info("=======================================================")
    log.info("         Image-to-Image Schrodinger Bridge")
    log.info("=======================================================")

    set_seed(opt.seed + opt.global_rank)

    # [MODIFIED] 根據 --use-single-dem 參數選擇數據集
    if opt.use_single_dem:
        log.info("Using single DEM flood dataset for training")
        train_dataset = singleDEMFloodDataset(opt, val=False, test=False)
        val_dataset   = singleDEMFloodDataset(opt, val=True, test=False)
    else:
        log.info("Using multi-DEM flood dataset for training")
        train_dataset = floodDataset(opt, val=False)
        val_dataset   = floodDataset(opt, val=True)

    rain_index = 0 if opt.use_single_dem else 1
    opt.train_rain_ids = sorted({p[rain_index] for p in train_dataset.cell_positions})
    opt.val_rain_ids = sorted({p[rain_index] for p in val_dataset.cell_positions})
    if not opt.train_rain_ids or not opt.val_rain_ids:
        raise ValueError('Train/validation split is empty after checking paired files.')
    if set(opt.train_rain_ids) & set(opt.val_rain_ids):
        raise ValueError('Rainfall event leakage between training and validation.')
    log.info(f'Train: {len(train_dataset)} samples, rainfall={opt.train_rain_ids}')
    log.info(f'Validation: {len(val_dataset)} samples, rainfall={opt.val_rain_ids}')
    os.makedirs(opt.ckpt_path, exist_ok=True)

    corrupt_method = build_corruption(opt, log)

    run = Runner(opt, log)

    # 只需要這次呼叫，不需要 DataLoader！
    run.train(opt, train_dataset, val_dataset, corrupt_method)

    log.info("Finish!")

if __name__ == '__main__':
    opt = create_training_options()

    assert opt.corrupt is not None

    # one-time download: ADM checkpoint
    # download_ckpt(DATA_DIR)

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # 嘗試使用 GPU（即使有 sm_120 警告）
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(opt.gpu if opt.gpu is not None else 0)
            except Exception as e:
                print(f"Warning: Could not set device - {e}")
        
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        init_processes(0, 1, main, opt)
