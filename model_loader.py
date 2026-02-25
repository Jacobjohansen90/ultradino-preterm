#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 10:17:10 2026

@author: jacob
"""

import ultradino_finetune.models.dinov2.load as vit_load
import logging
from models import EHRTransform, Predictor, BirthModel

logger = logging.getLogger("model_loader")

def vit_from_conf(cfg, **kwargs):
    # First create randomly initialized encoder
    model = vit_load.load_from_scratch(cfg.type, **kwargs)
    
    # Load pretrained weights if specified
    if cfg.weights_path:
        logger.info('Loading pretrained encoder from %s', cfg.weights_path)
        vit_load.load_pretrained_weights(model,
                                         cfg.weights_path)
    else:
        logger.info('No pretrained weights provided - encoder initialized randomly.')

    return model

def ehr_from_conf(cfg, **kwargs):
    #Currently dummy loader
    
    from torch import nn
    class PassThrough(nn.Module):
        def __init__(self):
            super(PassThrough, self).__init__()
            

        def forward(self, x):
            return x
    model = PassThrough()
    
    return model


def model_from_conf(cfg, **kwargs):
    """Create GA model from configuration"""

    #Possibility for kwargs to pretrained models - not currently used
    vit_kwargs = {}
    ehr_kwargs = {}

    device = kwargs['device'] if 'device' in kwargs else 'cpu'

    vit_model = vit_from_conf(cfg.vit, vit_kwargs)
    ehr_model = ehr_from_conf(cfg.ehr, ehr_kwargs)
    
    ehr_transform = EHRTransform(ehr_model.embed_dim, 
                                 cfg.ehr_transform.embedding_dim,
                                 layer_dims=cfg.ehr_transform.layer_dims)

    predictor = Predictor(vit_model.embed_dim,
                          cfg.predictor.layer_dims)

    model = BirthModel(vit_model,
                       ehr_model,
                       ehr_transform,
                       predictor,
                       aux_method=cfg.auxiliary.method)
    
    return model.to(device)

