#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:33:56 2026

@author: jacob
"""

from torch.utils.data import Dataset
import numpy as np

class DummyLoader(Dataset):
    def __init__(self, size=[224,224], scans=500):
        super().__init__()
        