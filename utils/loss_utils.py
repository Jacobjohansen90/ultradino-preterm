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
        
    mask = (labels*data['remove_on_GA']) != 0
                          
    return labels, mask
    
