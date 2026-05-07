#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:55:55 2026

@author: jacob
"""
import torch
import math
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(model, cfg):
    if cfg.optimizer.type == "AdamW":
        optim = torch.optim.AdamW(decay_lr(model, 
                                           base_lr=cfg.optimizer.lr,
                                           lr_decay=cfg.optimizer.lr_decay), 
                                  lr=cfg.optimizer.learning_rate,
                                  weight_decay=cfg.optimizer.weight_decay,
                                  betas=cfg.optimizer.adamw_params[0:2],
                                  eps=cfg.optimizer.adamw_params[2])

    elif cfg.optimizer.type == "Muon":
        optim = torch.optim.Muon(model.parameters(),
                                 lr=cfg.optimizer.learning_rate,
                                 weight_decay=cfg.optimizer.weight_decay)

    else:
        raise Exception(f"Optimizer {cfg.optimzier.type} not implemented")        
    
    return optim

def get_cosine_schedule_with_warmup(optimizer, conf, num_training_steps, last_epoch=-1):
    n_warmup_steps = conf.scheduler.num_warmup_steps
    vit_frozen = conf.training.vit_frozen_until
    num_cycles = conf.scheduler.num_cycles 
    epochs = conf.training.epochs

    def lr_lambda(current_step):
        if current_step < vit_frozen:
            x = float(current_step/vit_frozen)
            return abs(math.cos(math.pi*x))
        elif current_step < vit_frozen + n_warmup_steps:
            return float(current_step-vit_frozen) / float(n_warmup_steps)
        else:
            x = float((current_step-vit_frozen-n_warmup_steps)/epochs)*num_cycles
            return abs(math.cos(math.pi*x))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_layer_id(name):
    if name.startswith("patch_embed"):
        return 0

    elif name.startswith("blocks"):
        block_id = int(name.split(".")[1])
        return block_id + 1

    else:
        return 13
    
def decay_lr(model, base_lr, lr_decay):
    n_layers = 13  # patch_embed + 12 blocks

    param_groups = []

    for name, param in model.named_parameters():
        layer_id = get_layer_id(name)
        scale = lr_decay ** (n_layers - layer_id)
        param_groups.append({"params": [param],
                             "lr": base_lr * scale})

        print(name, base_lr * scale)

    return param_groups