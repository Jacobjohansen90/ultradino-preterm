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

logger = logging.getLogger("model_loader")

def vit_from_conf(conf, **kwargs):
    # First create randomly initialized encoder
    model = vit_load.load_from_scratch(conf.type, **kwargs)
    
    # Load pretrained weights if specified
    if conf.weights_path is not None:
        logger.info('Loading pretrained encoder from %s', conf.weights_path)
        vit_load.load_pretrained_weights(model,
                                         conf.weights_path)
    else:
        logger.info('No pretrained weights provided - encoder initialized randomly.')

    return model

def ehr_from_conf(conf, **kwargs):
    #Currently dummy loader
    
    from torch import nn
    class PassThrough(nn.Module):
        def __init__(self):
            super(PassThrough, self).__init__()
            self.embed_dim = 1
            

        def forward(self, x):
            return x
    model = PassThrough()
    
    return model


def model_from_conf(conf, **kwargs):
    """Create GA model from configuration"""

    #Possibility for kwargs to pretrained models - not currently used
    vit_kwargs = {}
    ehr_kwargs = {}

    device = conf.device.type

    vit_model = vit_from_conf(conf.model.vit, **vit_kwargs)
    ehr_model = ehr_from_conf(conf.model.ehr, **ehr_kwargs)
    
    img_data_transform = Transform(2, 
                                   vit_model.embed_dim)
    
    ehr_transform = Transform(ehr_model.embed_dim, 
                              vit_model.embed_dim,
                              layer_dims=conf.model.ehr_transform.layer_dims)

    predictor = FCPredictor(vit_model.embed_dim,
                            conf.model.predictor.layer_dims)

    model = BirthModel(vit_model,
                       ehr_model,
                       ehr_transform,
                       #img_data_transform,
                       predictor,
                       aux_method=conf.auxiliary.method)
    
    return model.to(device)

