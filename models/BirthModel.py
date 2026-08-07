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
                 preterm_heads,
                 aux_task_heads,
                 risk_transform=None,
                 aux_method='append'):
        
        super().__init__()
        
        self.ehr_model = ehr_model
        self.ehr_transform = ehr_transform
        self.risk_transform = risk_transform
        self.img_data_transform = img_data_transform
        self.vit_model = vit_model
        self.preterm_heads = preterm_heads
        self.aux_task_heads = aux_task_heads
        self.aux_method = aux_method
        
        if self.aux_method == 'append':
            """
            Early fusion, where EHR/Img-meta data is appended to the patch embeddings
            """
            self.forward_ = self.forward_append
        
        else:
            raise RuntimeError(f'Unknown fusion type f"{self.aux_method}"')

    def encode_ehr(self, ehr, patient_ids):
        if self.ehr_model is None:
            return None

        input_type = self.ehr_model.input_type

        if input_type == "patient_id":
            if patient_ids is None:
                return None
            return self.ehr_model(patient_ids)

        if input_type == "patient_id_tabular":
            if patient_ids is None or ehr.shape[1] == 0:
                return None
            return self.ehr_model(ehr, patient_ids)

        if ehr.shape[1] != 0:
            return self.ehr_model(ehr)

        return None

    def append_ehr_tokens(self, embeddings, ehr, patient_ids):
        encoded = self.encode_ehr(ehr, patient_ids)
        if encoded is None:
            return embeddings

        if self.ehr_model.input_type == "patient_id_tabular":
            risk, encoding = encoded
            if self.risk_transform is not None:
                embeddings.append(self.risk_transform(risk))
            if self.ehr_transform is not None:
                embeddings.append(self.ehr_transform(encoding))
            return embeddings

        if self.ehr_transform is not None:
            embeddings.append(self.ehr_transform(encoded))
        return embeddings
            
    def forward_append(self, img, img_data, ehr, patient_ids=None):
        embeddings = []

        self.append_ehr_tokens(embeddings, ehr, patient_ids)
        
        if img_data.shape[1] != 0:        
            img_data_embedding = self.img_data_transform(img_data)
            embeddings.append(img_data_embedding)
        
        if len(embeddings) > 0:
            embeddings = [torch.cat(embeddings, dim=1)] 
            vision_features = self.vit_model(img, append_tokens=embeddings)
        else:
            vision_features = self.vit_model(img)
        
        outputs = {'preterm': {},
                   'aux_tasks': {}}
        
        for GA, preterm_head in self.preterm_heads.items():
            outputs['preterm'][GA] = preterm_head(vision_features)
            
        for var, aux_task_head in self.aux_task_heads.items():
            outputs['aux_tasks'][var] = aux_task_head(vision_features)

        return outputs, vision_features
            
    def freeze_model(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def unfreeze_vit(self, model, n, cfg):
        total_blocks = len(model.blocks)
    
        #Always unfreeze final normalization layer
        for p in model.norm.parameters():
            p.requires_grad = True
    
        # Maximum number of blocks that are allowed to be trainable
        if cfg.top_n_blocks == -1:
            max_blocks = total_blocks
        else:
            max_blocks = cfg.top_n_blocks
    
        #Number of blocks to unfreeze
        if cfg.training.blocks_per_step == -1:
            n_blocks = max_blocks
        else:
            n_blocks = min(max_blocks, cfg.training.blocks_per_step * (n // cfg.training.every_n_epochs))
    
        if n_blocks > 0:
            for block in model.blocks[-n_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
    
        # Unfreeze embeddings once the whole backbone is trainable
        if n_blocks == total_blocks:
            model.cls_token.requires_grad = True
            model.pos_embed.requires_grad = True
            model.register_tokens.requires_grad = True

    def forward(self, img, img_data, ehr, patient_ids=None):
        return self.forward_(img, img_data, ehr, patient_ids)
