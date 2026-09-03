#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:30:20 2026

@author: jacob
"""

from omegaconf import OmegaConf
import os

def setup(cfg):
    if cfg.info.name is None:
        raise Exception("Model experiment must be named")
    
    path = f"/projects/users/data/UCPH/DeepFetal/projects/preterm/training_runs/Running/{cfg.info.name}/"
    if cfg.info.name != 'test':
        if os.path.exists(path):
            raise Exception("Model experiment exists in Running folder.")
        if os.path.exists(path.replace('Running', 'Evaluated')):
            raise Exception("Model experiment exists in Evaluated folder.")
            
    os.makedirs(path + 'misc', exist_ok=True)
    for i in range(cfg.data.folds):
        os.makedirs(path + 'weight/fold_' + str(i), exist_ok=True)
        
    OmegaConf.save(cfg, path + 'conf.yaml')        
    return path