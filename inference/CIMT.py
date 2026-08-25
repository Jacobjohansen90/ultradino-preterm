#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:30:43 2026

@author: jacob
"""

from utils.model_utils import model_from_conf
import torch

def inference(cfg, data, weight_path, cutoff):
    model = model_from_conf(cfg)
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()
    outputs, _ = model(data['imgs'].to(cfg.device.type),
                       data['img_data'].to(cfg.device.type),
                       data['ehr_data'].to(cfg.device.type))

    pred = outputs['preterm'][str(cutoff)]['preds'].flatten().cpu().numpy()
    
    return pred