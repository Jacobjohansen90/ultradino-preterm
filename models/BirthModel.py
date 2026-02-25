#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:46:47 2026

@author: jacob
"""

from torch import nn

class BirthModel(nn.Module):
    def __init__(self, 
                 vision_model, 
                 ehr_model, 
                 ehr_transform, 
                 predictor, 
                 aux_method='append'):
        
        super().__init__()
        
        self.ehr_model = ehr_model
        self.ehr_transform = ehr_transform
        self.vision_model = vision_model
        self.predictor = predictor
        self.aux_method = aux_method
        
        if self.aux_method == 'append':
            """
            Early fusion, where EHR info is appended to the patch embeddings
            """
            def forward(self, img, ehr):
                ehr_embedding = self.ehr_model(ehr)
                ehr_embedding = self.ehr_transform(ehr_embedding)
                vision_features = self.vision_model(img, appends_tokens=ehr_embedding)
                pred = predictor(vision_features)
                return pred
        else:
            raise RuntimeError(f'Unknown fusion type f"{self.fusion}"')