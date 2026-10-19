#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:08:41 2026

@author: jacob
"""
import torch 
from omegaconf import ListConfig

class ModelLoss():
    def __init__(self, cfg):
        self.loss_fns = get_loss(cfg)
        self.device = cfg.device.type
        tasks = {}

        for task_name, task_cfg in cfg.tasks.items():
            if task_name == "aux_tasks":
                tasks[task_name] = [[aux_cfg.var, aux_cfg.weights, aux_cfg.mask_value] for aux_cfg in task_cfg]
            
            elif task_name == 'preterm':
                cutoffs = task_cfg.get("cutoffs", [None])
                weights = task_cfg.get("weights", 1)

                if isinstance(weights, (int, float)):
                    weights = [weights] * len(cutoffs)
    
                tasks[task_name] = [[cutoff, weight, task_cfg.label_smoothing_value] for cutoff, weight in zip(cutoffs, weights)]
        
        self.tasks = tasks
        self.loss = {task_name: 0.0 for task_name in self.tasks.keys()}
        
    def __call__(self, outputs, data):
        loss = 0
        for task_name, task_values in self.tasks.items():
            if task_name == 'preterm':
                for cutoff, weight, ls_val in task_values:
                    labels = self.label_smoothing(data, cutoff, ls_val).to(self.device)
                    logits = outputs[task_name][str(cutoff)]['logits']
                    mask = data['masks'][str(cutoff)].to(self.device)
                    if mask.sum() > 0:
                        temp_loss = self.loss_fns[task_name](logits, labels)[mask].mean()*weight
                        loss += temp_loss
                        self.loss[task_name] += temp_loss.item() / mask.sum()

            elif task_name == 'segmentation':
                continue
            
            elif task_name == 'aux_tasks':
                for var, weight, mask_val in task_values:
                    labels = data['labels'][var].to(self.device)
                    logits = outputs[task_name][var]['logits']
                    mask = self.mask_values(labels, mask_val)
                    if mask.sum () > 0:
                        temp_loss = self.loss_fns[var](logits, labels)[mask].mean()*weight
                        loss += temp_loss
                        self.loss[task_name] += temp_loss.item() / mask.sum()

        return loss

    def get_loss(self, reset=True):
        loss_dict = self.loss
        if reset:
            self.loss = {task_name: 0.0 for task_name in self.tasks.keys()}

        return loss_dict

    def label_smoothing(self, data, cutoff, label_smoothing_value):    
        if label_smoothing_value > 0:
            labels = torch.sigmoid((cutoff-data['labels']['GA_weeks'])/label_smoothing_value)
        else:
            labels = (data['labels']['GA_weeks'] < cutoff).float()
        return labels
    
    def mask_values(self, labels, mask_value):
        if mask_value is None:
            return torch.ones_like(labels, dtype=torch.bool)
        else:
            return labels != mask_value
    
    
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
    loss_map = {'bce': torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(cfg.tasks.preterm.pos_weight,
                                                                                            dtype=torch.float32)),
                'l2': torch.nn.MSELoss(reduction='none'),
                'l1': torch.nn.L1Loss(reduction='none'),
                'dice': DiceLoss()}
    
    #Move to CUDA
    loss_map = {name: fn.to(cfg.device.type) for name, fn in loss_map.items()}
    
    losses = {}
    
    for task_name, config in cfg.tasks.items():
        tasks = config if isinstance(config, (list, ListConfig)) else [config]
        
        for task in tasks:            
            if task_name == 'aux_tasks':
                loss_name = task.var
            else:
                loss_name = task_name
                
            if isinstance(task['loss'], (list, ListConfig)):
                for name in task['loss']:
                    if name not in loss_map:
                        raise ValueError(f"Loss type '{name}' not implemented")
                
                def combined_loss(logits, labels, names=task['loss']):
                    result = 0
                    for name in names:
                        loss = loss_map[name](logits, labels)

                        # Reduce each component to one value per sample
                        if loss.ndim > 1:
                            loss = loss.flatten(1).mean(dim=1)

                        result = result + loss

                    return result.unsqueeze(1)

                losses[loss_name] = combined_loss
                
            else:
                if task['loss'] not in loss_map:
                    raise ValueError(f"Loss type '{task['loss']}' not implemented")
               
                losses[loss_name] = loss_map[task['loss']]
    
    
    return losses




    
