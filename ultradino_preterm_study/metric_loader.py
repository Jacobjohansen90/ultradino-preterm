#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:03:57 2026

@author: jacob
"""

from torchmetrics.classification import AUROC, Recall, Precision, Specificity, Accuracy

def get_metrics(device):
    metrics = {'AUC': AUROC(task='binary').to(device),
               'Recall': Recall(task='binary').to(device),
               'Precision': Precision(task='binary').to(device),
               'Specificity': Specificity(task='binary').to(device),
               'Accuracy': Accuracy(task='binary').to(device)}
    
    return metrics
    