#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:55:55 2026

@author: jacob
"""
import torch

def get_optimizer(model, conf):
    if conf.optimizer.type == "AdamW":
        optim = torch.optim.AdamW(model.parameters(), 
                                  lr=conf.optimizer.lr,
                                  weight_decay=conf.optimizer.lr.weight_decay,
                                  betas=conf.optimizer.adamw_params[0:2],
                                  eps=conf.optimizer.adamw_params[2])

    elif conf.optimizer.type == "Muon":
        optim = torch.optim.Muon(model.parameters(),
                                 lr=conf.optimizer.lr,
                                 weight_decay=conf.optimizer.lr.weight_decay)

    else:
        raise Exception(f"Optimizer {conf.optimzier.type} not implemented")        
    
    return optim