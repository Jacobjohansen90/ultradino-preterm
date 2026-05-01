#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:55:55 2026

@author: jacob
"""
import torch
import math
from torch.optim.lr_scheduler import LambdaLR

def get_optimizer(model, conf):
    if conf.optimizer.type == "AdamW":
        optim = torch.optim.AdamW(model.parameters(), 
                                  lr=conf.optimizer.learning_rate,
                                  weight_decay=conf.optimizer.weight_decay,
                                  betas=conf.optimizer.adamw_params[0:2],
                                  eps=conf.optimizer.adamw_params[2])

    elif conf.optimizer.type == "Muon":
        optim = torch.optim.Muon(model.parameters(),
                                 lr=conf.optimizer.learning_rate,
                                 weight_decay=conf.optimizer.weight_decay)

    else:
        raise Exception(f"Optimizer {conf.optimzier.type} not implemented")        
    
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

# def get_cosine_schedule_with_warmup(optimizer, conf, num_training_steps, last_epoch=-1):
#     """
#     Create a schedule with a learning rate that decreases following the values of the cosine function between the
#     initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
#     initial lr set in the optimizer.

#     Args:
#         optimizer:
#             The optimizer for which to schedule the learning rate.
#         conf:
#             Conf class from the yaml configuration file
#         num_training_steps:
#             The number of training steps per epoch
#         last_epoch:
#             The index of the last epoch when resuming training. (Defaults to -1)

#     Return:
#         LambdaLR function with the appropriate LR schedule.
#     """
#     num_warmup_steps = conf.scheduler.num_warmup_steps
#     cycles = conf.scheduler.num_cycles

#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
        
        
        
#         x = float((current_step-num_warmup_steps)/num_training_steps)
        
#         return abs(math.cos(math.pi*x*cycles))
        

#     return LambdaLR(optimizer, lr_lambda, last_epoch)
