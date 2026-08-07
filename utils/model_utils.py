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
from models.ehr_models import (
    TabularEhrModel,
    PatientIdEhrModel,
    PatientLookupTabularEhrModel,
)
from utils.ehr_encoding import load_ehr_encodings_from_cfg
import torch.nn as nn

logger = logging.getLogger("model_loader")

EHR_MODEL_TYPES = {
    "tabular": TabularEhrModel,
    "patient_lookup": PatientIdEhrModel,
    "patient_lookup_tabular": PatientLookupTabularEhrModel,
}


def vit_from_conf(cfg, **kwargs):
    if 'vitb16' in cfg.weights_path:
        model = vit_load.load_from_scratch('vitb16', **kwargs)
    elif 'vitl16'  in cfg.weights_path:
        model = vit_load.load_from_scratch('vitl16', **kwargs)
    else:
        raise RuntimeError(f"No model type found for {cfg.weights_path}")
        
    if cfg.weights_path is not None:
        logger.info('Loading pretrained encoder from %s', cfg.weights_path)
        vit_load.load_pretrained_weights(model, cfg.weights_path)
    else:
        logger.info('No pretrained weights provided - encoder initialized randomly.')
    
    set_dropout(model, cfg.dropout)
    
    return model


def resolve_ehr_model_type(cfg):
    ehr_cfg = cfg.model.get("ehr", {})
    if ehr_cfg.get("type"):
        return ehr_cfg.type
    if load_ehr_encodings_from_cfg(cfg):
        if cfg.data.get("ehr_data"):
            return "patient_lookup_tabular"
        return "patient_lookup"
    if cfg.data.get("ehr_data"):
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
            raise ValueError(
                "EHR model type 'tabular' requires data.ehr_data columns "
                "(from ehr_train_path/ehr_test_path or already in the parquet)"
            )
        return TabularEhrModel(len(cfg.data.ehr_data))

    encodings = load_ehr_encodings_from_cfg(cfg)
    if not encodings:
        raise ValueError(
            f"EHR model type '{model_type}' requires "
            "data.ehr_encoding_train_path / ehr_encoding_test_path"
        )

    encoding_dim = cfg.data.get("ehr_encoding_dim") or len(
        next(iter(encodings.values()))
    )
    logger.info(
        "Loaded %d patient encodings with dim %d",
        len(encodings),
        encoding_dim,
    )

    if model_type == "patient_lookup":
        return PatientIdEhrModel(encodings, encoding_dim)

    if not cfg.data.ehr_data:
        raise ValueError(
            "EHR model type 'patient_lookup_tabular' requires data.ehr_data "
            "risk columns (from ehr_train_path/ehr_test_path or the parquet)"
        )
    return PatientLookupTabularEhrModel(
        encodings, encoding_dim, len(cfg.data.ehr_data)
    )


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

    img_data_transform = Transform(len(cfg.data.img_data), 
                                   vit_model.embed_dim,
                                   layer_dims=cfg.model.transform.layer_dims)
    
    ehr_transform = None
    risk_transform = None
    if ehr_model is not None:
        ehr_cfg = cfg.model.get("ehr", {})
        ehr_num_tokens = ehr_cfg.get("num_tokens", 1)
        ehr_transform = Transform(ehr_model.embed_dim, 
                                  vit_model.embed_dim,
                                  layer_dims=cfg.model.transform.layer_dims,
                                  num_tokens=ehr_num_tokens)
        if ehr_model.input_type == "patient_id_tabular":
            risk_transform = Transform(ehr_model.tabular_dim,
                                       vit_model.embed_dim,
                                       layer_dims=cfg.model.transform.layer_dims,
                                       num_tokens=1)
    
    preterm_heads = nn.ModuleDict({})
    aux_task_heads = nn.ModuleDict({})
    
    for task in cfg.tasks.keys():
        if task == 'preterm':
            for cutoff in cfg.tasks[task].cutoffs:
                preterm_heads[str(cutoff)] = FCPredictor(vit_model.embed_dim,
                                                         cfg.model.head.dropout,
                                                         cfg.model.head.layer_dims)
        else:
            for aux_cfg in cfg.tasks[task]:
                aux_task_heads[aux_cfg['var']] = FCPredictor(vit_model.embed_dim,
                                                             cfg.model.head.dropout,
                                                             cfg.model.head.layer_dims)
        
    model = BirthModel(vit_model,
                       ehr_model,
                       ehr_transform,
                       img_data_transform,
                       preterm_heads,
                       aux_task_heads,
                       risk_transform=risk_transform,
                       aux_method=cfg.auxiliary.method)
    
    return model.to(device)


def update_freezing(model, epoch, cfg):
    if epoch == 0:
        model.freeze_model(model.vit_model)
        # Lookup/tabular ehr_models have no params (buffers / pass-through).
        # Learnable EHR path is ehr_transform / risk_transform.
        for transform in (model.ehr_transform, model.risk_transform):
            if transform is not None:
                model.freeze_model(transform)
        if model.ehr_model is not None and any(p.requires_grad for p in model.ehr_model.parameters()):
            model.freeze_model(model.ehr_model)
    if epoch >= cfg.training.vit_frozen_until:
        n = epoch - cfg.training.vit_frozen_until
        model.unfreeze_vit(model.vit_model, n, cfg)
    if epoch >= cfg.training.ehr_frozen_until:
        for transform in (model.ehr_transform, model.risk_transform):
            if transform is not None:
                for p in transform.parameters():
                    p.requires_grad = True
        if model.ehr_model is not None:
            for p in model.ehr_model.parameters():
                p.requires_grad = True
