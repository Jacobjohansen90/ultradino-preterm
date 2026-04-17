#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:35:53 2026

@author: jacob
"""

import csv
from torchmetrics.classification import BinarySensitivityAtSpecificity
import numpy as np
import math
import torch

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

csv_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/Joris/37w_data.csv"

with open(csv_path) as f:
    d = csv.reader(f)
    head = next(d)
    stats = {}
    
    for line in d:
        cpr = line[0]
        logit = line[1]
        label = line[2]
        fold = line[3]
        if cpr in stats.keys():
            if fold in stats[cpr]:
                stats[cpr][fold].append(logit)
            else:
                stats[cpr][fold] = [logit]
        else:
            stats[cpr] = {fold:[logit], 
                          'label':label}

#%% Avg logit

preds = []
labels = []
for key in stats.keys():
    avg_pred = []
    for fold in stats.keys():
        if fold == 'label':
            label = stats[key][fold]
        else:
            avg_pred.append(sigmoid(np.mean(stats[key][fold])))
    pred = np.mean(avg_pred)
    preds.append(pred)
    labels.append(label)
    
met = BinarySensitivityAtSpecificity(0.8)

print(met(torch.Tensor(preds), torch.Tensor(labels).to(int)))
    

        