#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:41:00 2026

@author: jacob
"""
#%%Imports
from omegaconf import OmegaConf, ListConfig
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from dataloader.dataloader import PreTermDataset, collate_fn, DataSplits
from utils.model_utils import model_from_conf, update_freezing
from utils.optim_loader import get_optimizer, get_cosine_schedule_with_warmup
from utils.loss_utils import get_loss, label_smoothing, mask_value
from utils.metrics import Metrics
from utils.utils import setup

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")

#%%Load config and setup logger(s) 
cfg = OmegaConf.load("/projects/users/data/UCPH/DeepFetal/projects/preterm/ultradino-preterm/confs/training_confs/append_tokens_vitb16.yaml")

save_path = setup(cfg)

#%% Load and train n folds of models

Data = DataSplits(cfg)
Data.save_distributions(save_path + 'misc/')

loss_fns = get_loss(cfg)
metrics = Metrics(cfg, save_path)

for fold in range(cfg.data.folds):

    train_df, test_df = Data.get_split(fold)
    
    TrainData = PreTermDataset(train_df, cfg, train=True)
    TestData = PreTermDataset(test_df, cfg, train=False)
    

    TrainLoader = DataLoader(TrainData,
                             cfg.data.batch_size,
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True,
                             num_workers=cfg.data.workers,
                             collate_fn=collate_fn)

    TestLoader = DataLoader(TestData,
                           cfg.data.batch_size,
                           shuffle=False,
                           pin_memory=False,
                           drop_last=False,
                           num_workers=cfg.data.workers,
                           collate_fn=collate_fn)

    model = model_from_conf(cfg)


    optimizer = get_optimizer(model, cfg)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg)


    for epoch in range(cfg.training.epochs):
        update_freezing(model, epoch, cfg)
    
        model.train()
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
                        labels = label_smoothing(data, cutoff, cfg.data.label_smoothing_param).to(cfg.device.type)
                        mask = data['masks'][cutoff].to(cfg.device.type)
                        if mask.sum() > 0:
                            preterm_loss = loss_fns[loss_fn](outputs[task][str(cutoff)]['logits'], labels)
                            loss += preterm_loss[mask].mean()*weight

                elif task == 'segmentation':
                    loss_fn, weight, _ = cfg.tasks[task].values()
                    labels = data['segmentation'].to(cfg.device.type)
                    if isinstance(loss_fn, (list, ListConfig)):
                        loss_fn = '_'.join(loss_fn)
                    mask = data['segmentation'].any(dim=(1, 2))
                    if mask.sum() > 0:
                        seg_loss = loss_fns[loss_fn](outputs['segmentation'], labels)
                        loss += seg_loss[mask].mean()*weight
            
                else:
                    for aux_task in cfg.tasks[task]:
                        var, loss_fn, weight, mask_val = aux_task.values()
                        labels = data['aux_vars'][var].to(cfg.device.type)
                        mask = mask_value(labels, mask_val)
                        if mask.sum() > 0:
                            aux_loss = loss_fns[loss_fn](outputs[task][var]['logits'], labels)
                            loss += aux_loss[mask].mean()*weight
                        
            loss.backward()
    
            train_loss += loss.item() / len(TrainLoader)
            optimizer.step()
            
        scheduler.step()
        
        model.eval()
        test_loss = 0
    
        with torch.no_grad():
            for data in TestLoader:
                outputs, _ = model(data['imgs'].to(cfg.device.type), 
                                   data['img_data'].to(cfg.device.type), 
                                   data['ehr_data'].to(cfg.device.type))
                
                metrics.update(outputs, data)
    
                loss = 0
                
                for task in cfg.tasks.keys():
                    if task == 'preterm':
                        cutoffs, loss_fn, weights = cfg.tasks[task].values()
                        for cutoff, weight in zip(cutoffs, weights):
                            labels = label_smoothing(data, cutoff, cfg.data.label_smoothing_param).to(cfg.device.type)
                            mask = data['masks'][cutoff].to(cfg.device.type)
                            if mask.sum() > 0:
                                preterm_loss = loss_fns[loss_fn](outputs[task][str(cutoff)]['logits'], labels)
                                loss += preterm_loss[mask].mean()*weight
    
                    elif task == 'segmentation':
                        loss_fn, weight, _ = cfg.tasks[task].values()
                        labels = data['segmentation'].to(cfg.device.type)
                        if isinstance(loss_fn, (list, ListConfig)):
                            loss_fn = '_'.join(loss_fn)
                        mask = data['segmentation'].any(dim=(1, 2))
                        if mask.sum() > 0:
                            seg_loss = loss_fns[loss_fn](outputs['segmentation'], labels)
                            loss += seg_loss[mask].mean()*weight
                
                    else:
                        for aux_task in cfg.tasks[task]:
                            var, loss_fn, weight, mask_val = aux_task.values()
                            labels = data['aux_vars'][var].to(cfg.device.type)
                            mask = mask_value(labels, mask_val)
                            if mask.sum() > 0:
                                aux_loss = loss_fns[loss_fn](outputs[task][var]['logits'], labels)
                                loss += aux_loss[mask].mean()*weight
    
                test_loss += loss.item() / len(TestLoader)
            
            metrics.log_metrics(train_loss, test_loss)
            torch.save(model.state_dict(), f"{save_path}/weights/fold_{fold}/{str(epoch).zfill(3)}.pth")        

    metrics.reset()

metrics.log_final_metrics()

