#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 15:30:06 2026

@author: jacob
"""
from torch import nn

from models.layers.FCLayer import FCLayer

class FCPredictor(nn.Module):
    def __init__(self, num_inputs, dropout, layer_dims=[]):
        super().__init__()
        
        self.num_inputs = num_inputs
        self.layer_dims = layer_dims

        layers = []
        last_dim = num_inputs
        
        for i in range(len(self.layer_dims)):
            layers.append(FCLayer(last_dim, self.layer_dims[i]))
            last_dim = self.layer_dims[i]
            
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(*layers)
        self.pred = nn.Linear(last_dim, 1)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        logits = self.pred(x)
        pred = self.act(logits)

        return logits, pred