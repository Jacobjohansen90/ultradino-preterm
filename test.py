#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:21:23 2026

@author: jacob
"""

from omegaconf import OmegaConf
from torch.utils.data import Subset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import os

from dataloader.dataloader import PreTermDataset, DummySet, collate_fn
from utils.model_loader import model_from_conf
from utils.optim_loader import get_optimizer, get_cosine_schedule_with_warmup
from utils.loss_loader import get_loss
from utils.metric_loader import get_metrics
from utils.documentation import setup, Logger


cfg = OmegaConf.load("/home/jacob/Desktop/NAS/Work/PreTerm/training_runs/2026-04-24 12:28:47/conf.yaml")

weights = "/home/jacob/Desktop/NAS/Work/PreTerm/training_runs/2026-04-24 12:28:47/weights/000.pth"
model = model_from_conf(cfg)

model.load_state_dict(torch.load(weights, weights_only=True))

cfg.data.path = "/projects/users/Data/test.json"
cutoff = 0.18
total = 0
score = 0
TestData = PreTermDataset(cfg, False)
TestLoader = DataLoader(TestData,
                        1,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False,
                        num_workers=8,
                        collate_fn=collate_fn)
for data in iter(TestLoader):
    logits, preds = model(data['img'].to(cfg.device.type), data['ehr_data'].to(cfg.device.type))
    pred = 1*(preds > cutoff)
    
    if pred == data['label']:
        score += 1
    total += 1
