#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:08:41 2026

@author: jacob
"""
import torch 
from omegaconf import ListConfig

def get_loss(cfg):
    losses = {}
    for config in cfg.tasks.values():
        if isinstance(config, (list, ListConfig)):
            for task in config:
                name = task['var']
                loss = task['loss']
        else:
            name = 'preterm'
            loss = config['loss']

        if loss == 'bce':
            losses[name] = torch.nn.BCEWithLogitsLoss(reduction="none")
        elif loss == 'l2':
            losses[name] = torch.nn.MSELoss()
        elif loss == 'l1':
            losses[name] = torch.nn.L1Loss() 
        else:
            raise Exception(f"Loss type {loss} not implemented")
        
    return losses


def fix_labels(data, cutoff, label_smoothing_param):
    if label_smoothing_param != 0:
        labels = torch.sigmoid((cutoff-data['GA_weeks'])/label_smoothing_param)
    else:
        labels = (data['GA_weeks'] < cutoff)
    labels = torch.tensor(labels, dtype=torch.int32)
    remove_on_GA = torch.tensor(data['remove_on_GA'], dtype=torch.int32)
    mask = (labels*remove_on_GA) != 0
                          
    return labels, mask
    
