from omegaconf import OmegaConf
import math
import random
import numpy as np
import torch

def get_params(path: str):
    
    """load parameters for experiment from yaml file"""

    params = OmegaConf.load(path)

    return params

def parse_optim_params(params):

    optim_params = {}

    if "lr" in params:
        optim_params["lr"] = params.lr
    
    if "weight_decay" in params:
        optim_params["weight_decay"] = params.weight_decay

    return optim_params

def adjust_learning_rate(optimizer, step, steps, warmup_steps, blr, min_lr=1e-6) -> None:
    """Decay the learning rate with half-cycle cosine after warmup"""
    if step < warmup_steps:
        lr = blr * step / warmup_steps 
    else:
        lr = min_lr + (blr - min_lr) * 0.5 * (1. + math.cos(math.pi * (step - warmup_steps) / (steps - warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

def seed_all(seed: int) -> None:
    "Set the random seed for reproducibility"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 