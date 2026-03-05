#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:46:47 2026

@author: jacob
"""

from torch import nn

class BirthModel(nn.Module):
    def __init__(self, 
                 vit_model, 
                 ehr_model, 
                 ehr_transform,
                 predictor,
                 aux_method='append'):
        
        super().__init__()
        
        self.ehr_model = ehr_model
        self.ehr_transform = ehr_transform
        self.vit_model = vit_model
        self.predictor = predictor
        self.aux_method = aux_method
        
        if self.aux_method == 'append':
            """
            Early fusion, where EHR info is appended to the patch embeddings
            """
            self.forward_ = self.forward_append
        
        else:
            raise RuntimeError(f'Unknown fusion type f"{self.fusion}"')
            
    def forward_append(self, img, ehr):
        ehr_embedding = self.ehr_model(ehr)
        ehr_embeddings = []
        for i in range(ehr_embedding.shape[1]):
            embedding = self.ehr_transform(ehr_embedding[:,i,:])
            ehr_embeddings.append(embedding.unsqueeze(1))
        vision_features = self.vit_model(img, append_tokens=ehr_embeddings)
        pred = self.predictor(vision_features)
        return pred
            
    def freeze_model(self, model):
        for n, p in model.named_parameters():
            p.requires_grad = False

    def unfreeze_model(self, model):
        for n, p in model.named_parameters():
            p.requires_grad = True

    def forward(self, img, ehr):
        return self.forward_(img, ehr)