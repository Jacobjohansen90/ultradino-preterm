#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 15:34:52 2026

@author: jacob
"""
from typing import Callable
from torch import nn

class FCLayer(nn.Module):
    def __init__(self, 
                 num_inputs, 
                 num_outputs, 
                 activation: Callable[..., nn.Module] = nn.GELU):
        
        super().__init__()
        
        self.FC = nn.Linear(num_inputs, num_outputs)
        self.act = activation
        
    def forward(self, x):
        x = self.FC(x)
        x = self.act(x)
        return x
        