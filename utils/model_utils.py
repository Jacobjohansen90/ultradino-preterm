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
from models.ehr_models import TabularEhrModel, PatientIdEhrModel
from utils.ehr_encoding import load_ehr_encodings_from_cfg
import torch.nn as nn

logger = logging.getLogger("model_loader")

EHR_MODEL_TYPES = {
    "tabular": TabularEhrModel,
    "patient_lookup": PatientIdEhrModel,
}


def vit_from_conf(cfg, **kwargs):
    model = vit_load.load_from_scratch(cfg.type, **kwargs)
    
    if cfg.weights_path is not None:
        logger.info('Loading pretrained encoder from %s', cfg.weights_path)
        vit_load.load_pretrained_weights(model,
                                         cfg.weights_path)
    else:
        logger.info('No pretrained weights provided - encoder initialized randomly.')
    
    set_dropout(model, cfg.dropout)
    
    return model


def resolve_ehr_model_type(cfg):
    ehr_cfg = cfg.model.get("ehr", {})
    if ehr_cfg.get("type"):
        return ehr_cfg.type
    if load_ehr_encodings_from_cfg(cfg):
        return "patient_lookup"
    if cfg.data.ehr_data:
        return "tabular"
    return None


def ehr_from_conf(cfg, **kwargs):
    model_type = resolve_ehr_model_type(cfg)

    if model_type is None:
        return None

    if model_type not in EHR_MODEL_TYPES:
        raise ValueError(
            f"Unknown EHR model type '{model_type}'. "
            f"Choose from {list(EHR_MODEL_TYPES)}"
        )

    if model_type == "tabular":
        if not cfg.data.ehr_data:
            raise ValueError("EHR model type 'tabular' requires data.ehr_data columns")
        return TabularEhrModel(len(cfg.data.ehr_data))

    encodings = load_ehr_encodings_from_cfg(cfg)
    if not encodings:
        raise ValueError(
            "EHR model type 'patient_lookup' requires "
            "data.ehr_encoding_train_path / ehr_encoding_test_path"
        )

    embed_dim = cfg.data.get("ehr_encoding_dim") or len(
        next(iter(encodings.values()))
    )
    logger.info(
        "Loaded %d patient encodings with dim %d",
        len(encodings),
        embed_dim,
    )
    return PatientIdEhrModel(encodings, embed_dim)


def set_dropout(model, p=0.1):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p


def model_from_conf(cfg, **kwargs):
    """Create GA model from configuration"""

    vit_kwargs = {}
    ehr_kwargs = {}

    device = cfg.device.type

    vit_model = vit_from_conf(cfg.model.vit, **vit_kwargs)
    ehr_model = ehr_from_conf(cfg, **ehr_kwargs)

    ehr_input_dim = ehr_model.embed_dim if ehr_model is not None else 0
    
    img_data_transform = Transform(len(cfg.data.img_data), 
                                   vit_model.embed_dim,
                                   layer_dims=cfg.model.transform.layer_dims)
    
    ehr_transform = None
    if ehr_input_dim > 0:
        ehr_transform = Transform(ehr_input_dim, 
                                  vit_model.embed_dim,
                                  layer_dims=cfg.model.transform.layer_dims)

    predictor = FCPredictor(vit_model.embed_dim,
                            cfg.model.pred_head.dropout,
                            cfg.model.pred_head.layer_dims)
    
    regressor = FCPredictor(vit_model.embed_dim,  
                            cfg.model.reg_head.dropout,
                            cfg.model.reg_head.layer_dims)

    model = BirthModel(vit_model,
                       ehr_model,
                       ehr_transform,
                       img_data_transform,
                       predictor,
                       regressor,
                       aux_method=cfg.auxiliary.method)
    
    model.freeze_model(model.vit_model)
    if ehr_model is not None:
        model.freeze_model(model.ehr_model)
    
    return model.to(device)


def freeze_model(model, epoch, cfg):
    if epoch >= cfg.training.vit_frozen_until:
        if cfg.training.strategy == 'all':
            model.unfreeze_model(model.vit_model)
    if epoch >= cfg.training.ehr_frozen_until:
        if cfg.training.strategy == 'all' and model.ehr_model is not None:
            model.unfreeze_model(model.ehr_model)
