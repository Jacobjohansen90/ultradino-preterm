#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:46:47 2026

@author: jacob
"""

import torch

from Pretrained.dinov2.load import load_from_scratch as load_vit
from Pretrained.ehr.load import load_from_scratch as load_ehr
from Models.Predictor import FCPredictor
from Models.CLSTransform import CLSTransform
from Models.BirthModel import BirthModel

def load_vit_from_state_dict(state_dict_path, device='cpu', return_dim=False):
    state_dict = torch.load(state_dict_path)
    embedding_size = state_dict['blocks.0.mlp.fc2.weight'].shape[0]

    if embedding_size == 768:
        model = load_vit('vitb16', deivice=device)
    elif embedding_size == 384:
        model = load_vit('vits16', device=device)
    else:
        raise RuntimeError(f'Unknown model type with embeddingsize f"{embedding_size}"')
    model.load_state_dict(state_dict)
    
    if return_dim:
        return model, embedding_size, state_dict['cls_token'].shape[-1]
    else:          
        return model

def load_ehr_from_state_dict(state_dict_path, device='cpu', return_dim=False):
    #Dummy loader for now

    state_dict = torch.load(state_dict_path)
    embedding_size, num_inputs = state_dict['fc.weight'].shape

    model = load_ehr(num_inputs, embedding_size, device=device)
    model.load_state_dict(state_dict)
   
    if return_dim:
        return model, embedding_size
    else:          
        return model

def load_model_from_scratch(vit_state_dict_path, 
                            ehr_state_dict_path,
                            pred_layer_dims=[],
                            cls_transform_dims=[],
                            device='cpu'):
    
    vit_model, vit_dim, cls_dim = load_vit_from_state_dict(vit_state_dict_path, device=device, return_dim=True)
    ehr_model, ehr_dim = load_ehr_from_state_dict(ehr_state_dict_path, device=device, return_dim=True)
    cls_transform = CLSTransform(ehr_dim, cls_dim)
    predictor = FCPredictor(vit_dim, pred_layer_dims)
    
    model = BirthModel(vit_model, ehr_model, cls_transform, predictor)
    
    return model.to(device)
    
