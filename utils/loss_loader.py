#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:08:41 2026

@author: jacob
"""
import torch 



def get_loss(cfg):
    losses = {}
    for name, loss in cfg.loss.tasks.items():
        if loss == 'bce':
            losses[name] = torch.nn.BCEWithLogitsLoss()
        elif loss == 'l2':
            losses[name] = torch.nn.MSELoss()
        elif loss == 'l1':
            losses[name] = torch.nn.L1Loss() 
        else:
            raise Exception(f"Loss type {cfg.loss.type} not implemented")
            
    return losses