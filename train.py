#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:41:00 2026

@author: jacob
"""

from omegaconf import OmegaConf
from torch.utils.data import Subset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

from dataloader.dataloader import PreTermDataset, DummySet
from utils.model_loader import model_from_conf
from utils.optim_loader import get_optimizer, get_cosine_schedule_with_warmup

conf = OmegaConf.load("./confs/append_tokens_vitb16.yaml")

#%% Setup dataloaders and models

TrainData = DummySet(train=True)
ValData = DummySet(train=False)

train_split, val_split = train_test_split(np.arange(len(TrainData)), 
                                          test_size=conf.data.val_frac)

TrainData = Subset(TrainData, train_split)
ValData = Subset(ValData, val_split)

#dataset = PreTermDataset(conf.data.path)


TrainLoader = DataLoader(TrainData,
                         conf.data.batch_size,
                         shuffle=True,
                         pin_memory=True,
                         drop_last=True,
                         num_workers=conf.data.workers)

ValLoader = DataLoader(ValData,
                       conf.data.batch_size,
                       shuffle=False,
                       pin_memory=False,
                       drop_last=False,
                       num_workers=conf.data.workers)


model = model_from_conf(conf)
model.freeze_model(model.vit_model)
model.freeze_model(model.ehr_model)

#%% Setup finetuning

optimizer = get_optimizer(model, conf)

scheduler = get_cosine_schedule_with_warmup(optimizer, conf, len(TrainLoader))

loss_fn = get_loss(conf)



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
        
