#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:30:43 2026

@author: jacob
"""

from utils.model_utils import model_from_conf
from omegaconf import OmegaConf
import torch

def load_model(cfg, weight_path):
    model = model_from_conf(cfg)
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()
    return model
    

def inference(model, data, cutoff, device_type='cude'):
    outputs, _ = model(data['imgs'].to(device_type),
                       data['img_data'].to(device_type),
                       data['ehr_data'].to(device_type))

    pred = outputs['preterm'][str(cutoff)]['preds'].flatten().cpu().numpy()
    
    return pred