#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 16:30:13 2026

@author: jacob
"""

from torch import nn

from models.layers.FCLayer import FCLayer
        
class Transform(nn.Module):
    def __init__(self, num_inputs, num_outputs, layer_dims=[], num_tokens=1):
        super().__init__()
        
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")

        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
        self.num_tokens = num_tokens
        self.layer_dims = list(layer_dims) + [num_outputs * num_tokens]

        layers = []
        last_dim = num_inputs
        
        for i in range(len(self.layer_dims)):
            layers.append(FCLayer(last_dim, self.layer_dims[i]))
            last_dim = self.layer_dims[i]
           
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x):
        # (B, 1, C) or (B, C) → (B, num_tokens, num_outputs)
        features = self.fc(x)
        if features.dim() == 2:
            features = features.unsqueeze(1)
        batch = features.shape[0]
        features = features.reshape(batch, self.num_tokens, self.num_outputs)
        return features
