#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:03:57 2026

@author: jacob
"""

import torchmetrics.classification as tm

def get_metrics(cfg, t=0.5):
    metrics = {'Recall': tm.Recall(task='binary', threshold=t).to(cfg.device.type),
               'Specificity': tm.Specificity(task='binary', threshold=t).to(cfg.device.type),
               'SensAtSpec': tm.SensitivityAtSpecificity(min_specificity=0.85, task='binary').to(cfg.device.type)}
    
    return metrics
    
