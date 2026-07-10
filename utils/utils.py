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
    
    path = f"/projects/users/data/UCPH/DeepFetal/projects/preterm/training_runs/Running/{cfg.info.name}_{str(cfg.data.ga_cutoff_weeks)}/"
    os.makedirs(path + 'weights/', exist_ok=False)
    if os.path.exists(path.replace('Running', 'Evaluated')):
        raise Exception("Model experiment exists in Evaluated folder.")
    OmegaConf.save(cfg, path + 'conf.yaml')        
    return path