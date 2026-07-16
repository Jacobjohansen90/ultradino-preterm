#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:32:36 2026

@author: jacob
"""

from utils.test_utils import test_model
import os

path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/training_runs/Evaluated/'

folders = os.listdir(path)

for folder in folders:
    print(f"Testing {folder}")
    test_model(path + folder + '/', move=False)
    print("Done")
    print()