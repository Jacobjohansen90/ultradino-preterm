#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:41:00 2026

@author: jacob
"""
#%%Imports
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from dataloader.dataloader import PreTermDataset, collate_fn, make_data_split
from utils.model_utils import model_from_conf, freeze_model
from utils.optim_loader import get_optimizer, get_cosine_schedule_with_warmup
from utils.loss_utils import get_loss, fix_labels
from utils.metrics import Metrics
from utils.utils import setup
from utils.test_utils import test_model

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")


#%%Load config and setup logger(s)
cfg = OmegaConf.load("/projects/users/data/UCPH/DeepFetal/projects/preterm/ultradino-preterm/confs/training_confs/append_tokens_vitb16.yaml")

save_path = setup(cfg)

#%% Setup dataloaders and models
train_df, val_df = make_data_split(cfg, cfg.data.path, unique_column='CPR_MOTHER')
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

train_population = TrainData.population_count(cfg.tasks.preterm.cutoffs)
val_population = ValData.population_count(cfg.tasks.preterm.cutoffs)

model = model_from_conf(cfg)

#%%Setup finetuning
optimizer = get_optimizer(model, cfg)
scheduler = get_cosine_schedule_with_warmup(optimizer, cfg, cfg.training.epochs)
loss_fns = get_loss(cfg)
metrics = Metrics(cfg, save_path)

for epoch in range(cfg.training.epochs):
    freeze_model(model, epoch, cfg)

    model.train(True)
    train_loss = 0.0
    for data in tqdm(TrainLoader):
        optimizer.zero_grad()
        outputs, _ = model(data['imgs'].to(cfg.device.type), 
                           data['img_data'].to(cfg.device.type), 
                           data['ehr_data'].to(cfg.device.type))
        loss = 0
        for task in cfg.tasks.keys():
            if task == 'preterm':
                cutoffs, loss_fn, weights = cfg.tasks[task].values()
                for cutoff, weight in zip(cutoffs, weights):
                    labels, mask = fix_labels(data, cutoff, cfg.data.label_smoothing_param)
                    mask = mask.to(cfg.device.type)
                    labels = labels.to(cfg.device.type)
                    preterm_loss = loss_fns[loss_fn](outputs[task][str(cutoff)]['logits'], labels)*weight
                    loss += (preterm_loss*mask).sum() / mask.sum().clamp(min=1)
            else:
                for aux_task in cfg.tasks[task]:
                    var, loss_fn, weight = aux_task.values()
                    labels = data[var].to(cfg.device.type)
                    loss += loss_fns[loss_fn](outputs[task][var]['logits'], labels)*weight
                    
        loss.backward()

        train_loss += loss.item() / len(TrainLoader)
        optimizer.step()
        
    scheduler.step()
    
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for data in iter(ValLoader):
            outputs, _ = model(data['imgs'].to(cfg.device.type), 
                               data['img_data'].to(cfg.device.type), 
                               data['ehr_data'].to(cfg.device.type))
            metrics.update(outputs, data)

            loss = 0
            
            for task in cfg.tasks.keys():
                if task == 'preterm':
                    cutoffs, loss_fn, weights = cfg.tasks[task].values()
                    for cutoff, weight in zip(cutoffs, weights):
                        labels, mask = fix_labels(data, cutoff, cfg.data.label_smoothing_param)
                        mask = mask.to(cfg.device.type)
                        labels = labels.to(cfg.device.type)     
                        preterm_loss = loss_fns[loss_fn](outputs[task][str(cutoff)]['logits'], labels)*weight
                        loss += (preterm_loss*mask).sum() / mask.sum().clamp(min=1)
                
                else:
                    for aux_task in cfg.tasks[task]:
                        var, loss_fn, weight = aux_task.values()
                        labels = data[var].to(cfg.device.type)
                        loss += loss_fns[loss_fn](outputs[task][var]['logits'], labels)*weight

            val_loss += loss.item() / len(ValLoader)
        
    metrics.log_metrics(train_loss, val_loss)
    torch.save(model.state_dict(), save_path + '/weights/' + str(epoch).zfill(3) + '.pth')        

#%%Test model and log results
test_model(save_path, cfg.data.test_path)
