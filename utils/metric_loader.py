#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:03:57 2026

@author: jacob
"""

from torchmetrics.classification import AUROC, Recall, Precision, Specificity, Accuracy, SensitivityAtSpecificity, SpecificityAtSensitivity

def get_metrics(cfg):
    # metrics = {'AUC': AUROC(task='binary').to(conf.device.type),
    #            'Recall': Recall(task='binary').to(conf.device.type),
    #            'Precision': Precision(task='binary').to(conf.device.type),
    #            'Specificity': Specificity(task='binary').to(conf.device.type),
    #            'Accuracy': Accuracy(task='binary').to(conf.device.type),
    #            'SensAtSpec': SensitivityAtSpecificity(min_specificity=0.85, task='binary').to(conf.device.type),
    #            'SpecAtSens': SpecificityAtSensitivity(min_sensitivity=0.7, task='binary').to(conf.device.type)}
    
    metrics = {'Recall': Recall(task='binary').to(cfg.device.type),
               'Specificity': Specificity(task='binary').to(cfg.device.type),
               'SensAtSpec': SensitivityAtSpecificity(min_specificity=0.85, task='binary').to(cfg.device.type)}
    
    return metrics
    
def get_test_metrics(cfg, t):
    
    metrics = {'Sens': Recall(task='binary', threshold=t).to(cfg.device.type),
               'Spec': Specificity(task='binary', threshold=t).to(cfg.device.type),
               'SensAtSpec': SensitivityAtSpecificity(min_specificity=0.85, task='binary').to(cfg.device.type)}
    
    return metrics