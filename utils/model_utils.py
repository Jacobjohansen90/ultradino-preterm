#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 10:17:10 2026

@author: jacob
"""

import ultradino_finetune.models.dinov2.load as vit_load
import logging
from models.Transform import Transform
from models.Predictor import FCPredictor
from models.BirthModel import BirthModel
import torch.nn as nn

logger = logging.getLogger("model_loader")

def vit_from_conf(cfg, **kwargs):
    # First create randomly initialized encoder
    if 'vitb16' in cfg.weights_path:
        model = vit_load.load_from_scratch('vitb16', **kwargs)
    elif 'vitl16'  in cfg.weights_path:
        model = vit_load.load_from_scratch('vitl16', **kwargs)
    else:
        raise Exception(f"No model type found for {cfg.weights_path}")
        
    
    # Load pretrained weights if specified
    if cfg.weights_path is not None:
        logger.info('Loading pretrained encoder from %s', cfg.weights_path)
        vit_load.load_pretrained_weights(model, cfg.weights_path)
    else:
        logger.info('No pretrained weights provided - encoder initialized randomly.')
    
    set_dropout(model, cfg.dropout)
    
    return model

def ehr_from_conf(cfg, **kwargs):
    #Currently dummy model    
    from torch import nn
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

def model_from_conf(cfg, **kwargs):
    """Create GA model from configuration"""

    #Possibility for kwargs to pretrained models - not currently used
    vit_kwargs = {}
    ehr_kwargs = {}

    device = cfg.device.type

    vit_model = vit_from_conf(cfg.model.vit, **vit_kwargs)
    ehr_model = ehr_from_conf(cfg, **ehr_kwargs)
    
    img_data_transform = Transform(len(cfg.data.img_data), 
                                   vit_model.embed_dim,
                                   layer_dims=cfg.model.transform.layer_dims)
    
    ehr_transform = Transform(len(cfg.data.ehr_data), 
                              vit_model.embed_dim,
                              layer_dims=cfg.model.transform.layer_dims)
    
    preterm_heads = nn.ModuleDict({})
    aux_task_heads = nn.ModuleDict({})
    
    for task in cfg.tasks.keys():
        if task == 'preterm':
            for cutoff in cfg.tasks[task].cutoffs:
                preterm_heads[str(cutoff)] = FCPredictor(vit_model.embed_dim,
                                                         cfg.model.head.dropout,
                                                         cfg.model.head.layer_dims)
            else:
                for task in cfg.tasks[task]:
                    aux_task_heads[task['var']] = FCPredictor(vit_model.embed_dim,
                                                              cfg.model.head.dropout,
                                                              cfg.model.head.layer_dims)
        
    model = BirthModel(vit_model,
                       ehr_model,
                       ehr_transform,
                       img_data_transform,
                       preterm_heads,
                       aux_task_heads,
                       aux_method=cfg.auxiliary.method)
    
    model.freeze_model(model.vit_model)
    model.freeze_model(model.ehr_model)
    
    return model.to(device)

def freeze_model(model, epoch, cfg):
    if epoch >= cfg.training.vit_frozen_until:
        if cfg.training.strategy == 'all':
            model.unfreeze_model(model.vit_model)
    if epoch >= cfg.training.ehr_frozen_until:
        if cfg.training.strategy == 'all':
            model.unfreeze_model(model.ehr_model)
            
