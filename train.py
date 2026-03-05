#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:41:00 2026

@author: jacob
"""

from omegacli import OmegaConf
from torch.utils.data import random_split, DataLoader
import torch.optim as optim

from dataloader.dataloader import PreTermDataset, DummySet
from utils.model_loader import model_from_conf

conf = OmegaConf.load("./confs/append_tokens_vitb16.yaml")

#%% Setup dataloaders and models

dataset = DummySet()
#dataset = PreTermDataset(conf.data.path)

train_ds, val_ds = random_split(dataset, [1-conf.data.val_frac, conf.data.val_frac])

TrainLoader = DataLoader(train_ds, 
                         conf.data.batch_size,
                         shuffle=True,
                         pin_memory=True,
                         num_workers=conf.data.workers)

ValLoader = DataLoader(val_ds,
                       conf.data.batch_size,
                       shuffle=False,
                       pin_memory=False,
                       num_workers=conf.data.workers)


model = model_from_conf(conf)
model.freeze_model(model.vit_model)
model.freeze_model(model.ehr_model)

#%% Setup finetuning

optimizer = optim.AdamW()

for epoch in range(conf.training.epochs):
    if epoch == conf.training.vit_frozen_until:
        model.unfreeze_model(model.vit_model)
        
    if epoch == conf.training.ehr_frozen_until:
        model.unfreeze_model(model.ehr_model)

    model.train(True)
    
    for data in iter(TrainLoader):
        optimizer.zero_grad()
        outputs = model(data['img'], data['ehr_data'])
        loss = loss_fn(outputs, data['labels'])
        loss.backward()
        
        optimizer.step()
        
