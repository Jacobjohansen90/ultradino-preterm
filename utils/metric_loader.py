#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:03:57 2026

@author: jacob
"""

from torchmetrics.classification import AUROC, Recall, Precision, Specificity, Accuracy, SensitivityAtSpecificity

def get_metrics(conf):
    metrics = {'AUC': AUROC(task='binary').to(conf.device.type),
               'Recall': Recall(task='binary').to(conf.device.type),
               'Precision': Precision(task='binary').to(conf.device.type),
               'Specificity': Specificity(task='binary').to(conf.device.type),
               'Accuracy': Accuracy(task='binary').to(conf.device.type),
               'SensAtSpec': SensitivityAtSpecificity(min_specificity=0.85, task='binary').to(conf.device.type)}
    
    return metrics
    