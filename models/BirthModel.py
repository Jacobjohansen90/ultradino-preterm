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
                 predictor,
                 regressor,
                 aux_method='append',
                 aux_strategy='sum'):
        
        super().__init__()
        
        self.ehr_model = ehr_model
        self.ehr_transform = ehr_transform
        self.img_data_transform = img_data_transform
        self.vit_model = vit_model
        self.predictor = predictor
        self.regressor = regressor
        self.aux_method = aux_method
        self.aux_strategy = aux_strategy
        
        if self.aux_method == 'append':
            """
            Early fusion, where EHR info is appended to the patch embeddings
            """
            self.forward_ = self.forward_append
        
        else:
            raise RuntimeError(f'Unknown fusion type f"{self.fusion}"')
            
    def forward_append(self, img, img_data, ehr):
        print(ehr)
        ehr_embedding = self.ehr_model(ehr)
        ehr_embeddings = []
        for i in range(ehr_embedding.shape[1]):
            embedding = self.ehr_transform(ehr_embedding[:,i,:])
            ehr_embeddings.append(embedding)
        ehr_embeddings = torch.cat(ehr_embeddings, dim=1)
        
        img_data_embeddings = []
        for i in range(img_data.shape[1]):
            img_data_embedding = self.img_data_transform(img_data[:,i,:])
            img_data_embeddings.append(img_data_embedding)
        img_data_embeddings = torch.cat(img_data_embeddings, dim=1)
        
        if self.aux_strategy == 'sum':
            img_data_embeddings = img_data_embeddings.sum(dim=1, keepdim=True)
            ehr_embeddings = ehr_embeddings.sum(dim=1, keepdim=True)

            
        embeddings = [torch.cat((img_data_embeddings, ehr_embeddings), dim=1)]
 
        vision_features = self.vit_model(img, append_tokens=embeddings)
        
        GA_reg, _ = self.regressor(vision_features)
        preterm_logits, preterm_pred = self.predictor(vision_features)
        
        return {'preterm': preterm_pred,
                'preterm_logits': preterm_logits,
                'vision_features': vision_features,
                'GA_reg': GA_reg}
            
    def freeze_model(self, model):
        for n, p in model.named_parameters():
            p.requires_grad = False

    def unfreeze_model(self, model):
        for n, p in model.named_parameters():
            p.requires_grad = True

    def forward(self, img, img_data, ehr):
        return self.forward_(img, img_data, ehr) 
    
