#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:46:47 2026

@author: jacob
"""

from torch import nn
import torch

class BirthModel(nn.Module):
    def __init__(self, 
                 vit_model, 
                 ehr_model, 
                 ehr_transform,
                 img_data_transform,
                 predictors,
                 regressors,
                 aux_method='append',
                 aux_strategy='sum'):
        
        super().__init__()
        
        self.ehr_model = ehr_model
        self.ehr_transform = ehr_transform
        self.img_data_transform = img_data_transform
        self.vit_model = vit_model
        self.predictors = predictors
        self.regressors = regressors
        self.aux_method = aux_method
        self.aux_strategy = aux_strategy
        
        if self.aux_method == 'append':
            """
            Early fusion, where EHR/Img-meta data is appended to the patch embeddings
            """
            self.forward_ = self.forward_append
        
        else:
            raise RuntimeError(f'Unknown fusion type f"{self.fusion}"')
            
    def forward_append(self, img, img_data, ehr):
        embeddings = []
        if ehr.shape[1] != 0:
            ehr_embedding = self.ehr_model(ehr)        
            ehr_embedding = self.ehr_transform(ehr_embedding)
            embeddings.append(ehr_embedding)
        
        if img_data.shape[1] != 0:        
            img_data_embedding = self.img_data_transform(img_data)
            embeddings.append(img_data_embedding)
        
        if len(embeddings) > 0:
            embeddings = [torch.cat(embeddings, dim=1)] 
            vision_features = self.vit_model(img, append_tokens=embeddings)
        else:
            vision_features = self.vit_model(img)
        
        outputs = {'preterm': {},
                   'regression': {}}
        
        for GA, predictor in self.predictors:
            outputs['preterm'][GA] = predictor(vision_features)
            
        for var, regressor in self.regressors:
            outputs['regression'][var] = regressor(vision_features)

        return outputs, vision_features
            
    def freeze_model(self, model):
        for n, p in model.named_parameters():
            p.requires_grad = False

    def unfreeze_model(self, model):
        for n, p in model.named_parameters():
            p.requires_grad = True

    def forward(self, img, img_data, ehr):
        return self.forward_(img, img_data, ehr) 
    
