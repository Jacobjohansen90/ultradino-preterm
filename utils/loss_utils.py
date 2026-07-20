#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:08:41 2026

@author: jacob
"""
import torch 
from omegaconf import ListConfig

def get_loss(cfg):
    loss_map = {'bce': torch.nn.BCEWithLogitsLoss(reduction='none'),
                'l2': torch.nn.MSELoss(reduction='mean'),
                'l1': torch.nn.L1Loss(reduction='mean')}    
    
    losses = {}
    
    for config in cfg.tasks.values():
        tasks = config if isinstance(config, (list, ListConfig)) else [config]
        
        for task in tasks:
            loss_name = task['loss']
            
            if loss_name not in loss_map:
                raise ValueError(f"Loss type '{loss_name}' not implemented")
                
            else:
                losses[loss_name] = loss_map[loss_name]
    
    return losses


def fix_labels(data, cutoff, label_smoothing_param):
    
    positive = (data['GA_weeks'] < cutoff).bool()
    remove_on_GA = (data['remove_on_GA']).bool()
    
    if label_smoothing_param != 0:
        labels = torch.sigmoid((cutoff-data['GA_weeks'])/label_smoothing_param)
    else:
        labels = positive.float()
        
    mask = ~(positive & remove_on_GA)

    return labels, mask
    
