#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:08:41 2026

@author: jacob
"""
import torch 



def get_loss(cfg):
    losses = {}
    for name, (loss, weight) in cfg.labels.tasks.items():
        if loss == 'bce':
            losses[name] = torch.nn.BCELoss()
        elif loss == 'l2':
            losses[name] = torch.nn.MSELoss()
        elif loss == 'l1':
            losses[name] = torch.nn.L1Loss() 
        else:
            raise Exception(f"Loss type {loss} not implemented")
            
    return losses