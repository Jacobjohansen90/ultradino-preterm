#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 16:30:13 2026

@author: jacob
"""

from torch import nn

from models.layers.FCLayer import FCLayer
        
class EHRTransform(nn.Module):
    def __init__(self, num_inputs, num_outputs, layer_dims=[]):
        super().__init__()
        
        self.num_ouputs = num_outputs
        self.num_inputs = num_inputs
        self.layer_dims = layer_dims + [num_outputs]

        layers = []
        last_dim = num_inputs
        
        for i in range(len(self.layer_dims)):
            layers.append(FCLayer(last_dim, self.layer_dims[i]))
            last_dim = self.layer_dims[i]
           
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x):
        features = self.fc(x)

        return features        
