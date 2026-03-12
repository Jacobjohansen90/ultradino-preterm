#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:08:41 2026

@author: jacob
"""
import torch 

def get_loss(conf):
    if conf.loss.type == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise Exception(f"Loss type {conf.loss.type} not implemented")