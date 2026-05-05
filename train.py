#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:41:00 2026

@author: jacob
"""

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os

from dataloader.dataloader import PreTermDataset, DummySet, collate_fn, make_train_val_split
from utils.model_loader import model_from_conf
from utils.optim_loader import get_optimizer, get_cosine_schedule_with_warmup
from utils.loss_loader import get_loss
from utils.metric_loader import get_metrics
from utils.documentation import setup, Logger

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")

if 'Jacob' in os.uname()[1]:
    cfg = OmegaConf.load("/home/jacob/Desktop/NAS/Work/PreTerm/ultradino-preterm/confs/training_confs/append_tokens_vitb16.yaml")
    cfg.model.vit.weights_path = None
else:
    cfg = OmegaConf.load("/projects/users/data/UCPH/DeepFetal/projects/preterm/ultradino-preterm/confs/training_confs/append_tokens_vitb16.yaml")

save_path = setup(cfg)

logger = Logger(save_path)

#%% Setup dataloaders and models

if 'Jacob' in os.uname()[1]:
    TrainData = DummySet(train=True, scans=500)
    ValData = DummySet(train=False, scans=500)
else:
    train_df, val_df = make_train_val_split(cfg, unique_column='CPR_MOTHER')
    TrainData = PreTermDataset(train_df, cfg, train=True)
    ValData = PreTermDataset(val_df, cfg, train=False)
    

TrainLoader = DataLoader(TrainData,
                         cfg.data.batch_size,
                         shuffle=True,
                         pin_memory=True,
                         drop_last=True,
                         num_workers=cfg.data.workers,
                         collate_fn=collate_fn)

ValLoader = DataLoader(ValData,
                       cfg.data.batch_size,
                       shuffle=False,
                       pin_memory=False,
                       drop_last=False,
                       num_workers=cfg.data.workers,
                       collate_fn=collate_fn)


model = model_from_conf(cfg)
model.freeze_model(model.vit_model)
model.freeze_model(model.ehr_model)

#%% Setup finetuning

optimizer = get_optimizer(model, cfg)
scheduler = get_cosine_schedule_with_warmup(optimizer, cfg, cfg.training.epochs)
loss_fns = get_loss(cfg)
metrics = get_metrics(cfg)

for epoch in range(cfg.training.epochs):
    if epoch == cfg.training.vit_frozen_until:
        model.unfreeze_model(model.vit_model)
        
    if epoch == cfg.training.ehr_frozen_until:
        model.unfreeze_model(model.ehr_model)

    model.train(True)
    train_loss = 0
    for i, data in enumerate(tqdm(TrainLoader)):
        optimizer.zero_grad()
        outputs = model(data['img'].to(cfg.device.type), 
                        data['img_data'].to(cfg.device.type), 
                        data['ehr_data'].to(cfg.device.type))
        
        loss = 0
        for task, loss_fn in loss_fns.items():
            loss += loss_fn(outputs[task][0], data['label'][task].to(cfg.device.type))
        
        # loss = loss_fn(logits, data['label'].to(cfg.device.type))
        
        loss.backward()

        train_loss += loss.item() / len(TrainLoader)
        optimizer.step()
        
    scheduler.step()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in iter(ValLoader):
            outputs = model(data['img'].to(cfg.device.type), 
                            data['img_data'].to(cfg.device.type), 
                            data['ehr_data'].to(cfg.device.type))
            
            labels = data['label']['cls'].to(cfg.device.type)
            loss = loss_fn(outputs['cls'][0], labels)
            val_loss += loss.item() / len(ValLoader)
            for key in metrics.keys():
                metrics[key](outputs['cls'][1], labels.to(torch.int))
    

    torch.save(model.state_dict(), save_path + '/weights/' + str(epoch).zfill(3) + '.pth')        
    
    logger.log_metrics(metrics, train_loss, val_loss)
    logger.plot_metrics()
   
        
