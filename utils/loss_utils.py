#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:08:41 2026

@author: jacob
"""
import torch 
from omegaconf import ListConfig

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)

        # Flatten per sample
        probs = probs.flatten(1)
        target = target.float().flatten(1)

        intersection = (probs * target).sum(dim=1)
        denominator = probs.sum(dim=1) + target.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)

        return 1 - dice

def get_loss(cfg):
    loss_map = {'bce': torch.nn.BCEWithLogitsLoss(reduction='none'),
                'l2': torch.nn.MSELoss(reduction='none'),
                'l1': torch.nn.L1Loss(reduction='none'),
                'dice': DiceLoss()}    
    
    losses = {}
    
    for config in cfg.tasks.values():
        tasks = config if isinstance(config, (list, ListConfig)) else [config]
        
        for task in tasks:
            loss_name = task['loss']
            
            if isinstance(loss_name, (list, ListConfig)):
                for name in loss_name:
                    if loss_name not in loss_map:
                        raise ValueError(f"Loss type '{loss_name}' not implemented")
                
                def combined_loss(input, target, names=loss_name):
                    result = 0

                    for name in names:
                        loss = loss_map[name](input, target)

                        # Reduce each component to one value per sample
                        if loss.ndim > 1:
                            loss = loss.flatten(1).mean(dim=1)

                        result = result + loss

                    return result

                losses['_'.join(loss_name)] = combined_loss
                
            else:
                if loss_name not in loss_map:
                    raise ValueError(f"Loss type '{loss_name}' not implemented")
               
                losses[loss_name] = loss_map[loss_name]
    
    return losses

def get_mask(labels, mask_value):
    if mask_value is None:
        return torch.ones_like(labels, dtype=torch.bool)
    else:
        return labels != mask_value

def fix_labels(data, cutoff, label_smoothing_param):
    
    positive = (data['GA_weeks'] < cutoff)
    remove_on_GA = data['remove_on_GA']
    
    if label_smoothing_param > 0:
        labels = torch.sigmoid((cutoff-data['GA_weeks'])/label_smoothing_param)
    else:
        labels = positive.float()
        
    mask = ~(positive & remove_on_GA)

    return labels, mask
    
