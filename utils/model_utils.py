#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 10:17:10 2026

@author: jacob
"""

import ultradino_finetune.models.dinov2.load as vit_load
from ultradino_segmentation import UltraDINOSegmentationModel
import logging
from models.Transform import Transform, CNNTransform
from models.Predictor import FCPredictor
from models.BirthModel import BirthModel
import torch.nn as nn
import re

logger = logging.getLogger("model_loader")

def vit_from_conf(cfg):    
    match = re.search(r'vit\w{3}', cfg.model.vit.weights_path)
    model_type = match.group() if match else None

    if model_type is not None:
        model = vit_load.load_pretrained_ultradino(model_type, cfg.model.vit.weights_path)
        logger.info('Loading pretrained encoder from %s', cfg.model.vit.weights_path)
    else:
        raise Exception("Weight path must contain the model type [Vits16/Vitb16/Vitl16]")
            
    return model


# def seg_vit_from_conf(cfg, **kwargs):
#     if 'vitb16' in cfg.weights_path:
#         logger.info('Loading pretrained encoder from %s', cfg.weights_path)
#         model = UltraDINOSegmentationModel.from_pretrained(model_type='vitb16',
#                                                            weights_path=cfg.weights_path,
#                                                            num_classes=1)
#     else:
#         raise Exception(f"Segmentation model does not support model type from weights path {cfg.weights_path}")
#     return model

def ehr_from_conf(cfg, **kwargs):
    #Placeholder for EHR model
    #Remember to update freezeing in BirthModel to accomedate EHR model if implemented!
    class PassThrough(nn.Module):
        def __init__(self):
            super(PassThrough, self).__init__()
            self.embed_dim = len(cfg.data.ehr_data)
            
        def forward(self, x):
            return x
    model = PassThrough()
    
    return model

def set_dropout(model, p=0.1):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p

def model_from_conf(cfg):
    """Create GA model from configuration"""
    
    device = cfg.device.type
    
    model_parts = {}
    
    #Foundation models
    vit_model = vit_from_conf(cfg)
    ehr_model = ehr_from_conf(cfg)
    
    model_parts['vit_model'] = vit_model
    model_parts['ehr_model'] = ehr_model
    
    #Transforms
    img_data_transform = Transform(len(cfg.data.img_data), 
                                   vit_model.embed_dim,
                                   layer_dims=cfg.model.transform.layer_dims)
    
    ehr_transform = Transform(ehr_model.embed_dim, 
                              vit_model.embed_dim,
                              layer_dims=cfg.model.transform.layer_dims)
        
    segmentation_transform = CNNTransform(cfg.data.img_size, 
                                          vit_model.embed_dim,
                                          layer_dims=cfg.model.transform.layer_dims)
    
    model_parts['img_data_transform'] = img_data_transform
    model_parts['ehr_transform'] = ehr_transform
    model_parts['segmentation_transform'] = segmentation_transform

    #Predictor heads
    preterm_heads = nn.ModuleDict({})
    aux_task_heads = nn.ModuleDict({})
    
    for task in cfg.tasks.keys():
        if task == 'preterm':
            for cutoff in cfg.tasks[task].cutoffs:
                preterm_heads[str(cutoff)] = FCPredictor(vit_model.embed_dim,
                                                         cfg.model.head.dropout,
                                                         cfg.model.head.layer_dims)
                
        elif task == 'aux_tasks':
            for aux_cfg in cfg.tasks[task]:
                aux_task_heads[aux_cfg['var']] = FCPredictor(vit_model.embed_dim,
                                                             cfg.model.head.dropout,
                                                             cfg.model.head.layer_dims)

    model_parts['preterm_heads'] = preterm_heads
    model_parts['aux_task_heads'] = aux_task_heads

    
    model = BirthModel(model_parts, aux_method=cfg.auxiliary.method)
    
    return model.to(device)

def update_freezing(model, epoch, cfg):
    if epoch == 0:
        model.freeze_model(model.vit_model)
        model.freeze_model(model.ehr_model)
    if epoch >= cfg.training.vit_frozen_until:
        n = epoch - cfg.training.vit_frozen_until
        model.unfreeze_vit(model.vit_model, n, cfg)

            
