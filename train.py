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

from dataloader.dataloader import PreTermDataset, collate_fn, make_train_val_split
from utils.model_utils import model_from_conf, freeze_model
from utils.optim_loader import get_optimizer, get_cosine_schedule_with_warmup
from utils.loss_loader import get_loss
from utils.metric_loader import get_metrics
from utils.documentation import setup, Logger

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")

cfg = OmegaConf.load("/projects/users/data/UCPH/DeepFetal/projects/preterm/ultradino-preterm/confs/training_confs/append_tokens_vitb16.yaml")

save_path = setup(cfg)

logger_avg = Logger(save_path, name='avg')
logger_max = Logger(save_path, name='max')

#%% Setup dataloaders and models


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
                       1,
                       shuffle=False,
                       pin_memory=False,
                       drop_last=False,
                       num_workers=cfg.data.workers,
                       collate_fn=collate_fn)


model = model_from_conf(cfg)

#%% Setup finetuning

optimizer = get_optimizer(model, cfg)
scheduler = get_cosine_schedule_with_warmup(optimizer, cfg, cfg.training.epochs)
loss_fns = get_loss(cfg)

metrics_avg = get_metrics(cfg)
metrics_max = get_metrics(cfg)

for epoch in range(cfg.training.epochs):
    freeze_model(model, epoch, cfg)

    model.train(True)
    train_loss = 0
    for i, data in enumerate(tqdm(TrainLoader)):
        optimizer.zero_grad()
        outputs = model(data['img'].to(cfg.device.type), 
                        data['img_data'].to(cfg.device.type), 
                        data['ehr_data'].to(cfg.device.type))
        
        loss = 0
        for task, (_, weight) in cfg.labels.tasks.items():
            loss += loss_fns[task](outputs[task], data['labels'][task].to(cfg.device.type))*weight
            
        loss.backward()

        train_loss += loss.item() / len(TrainLoader)
        optimizer.step()
        
    scheduler.step()
    model.eval()

    val_loss_avg = 0
    val_loss_max = 0

    with torch.no_grad():
        for data in iter(ValLoader):
            outputs = model(data['img'].to(cfg.device.type), 
                            data['img_data'].to(cfg.device.type), 
                            data['ehr_data'].to(cfg.device.type))
            
            output_avg = outputs['preterm'].mean()
            output_max = outputs['preterm'].max()
            label = data['labels']['preterm'][0].to(cfg.device.type)[0]
                        
            loss_avg = loss_fns['preterm'](output_avg, label)
            loss_max = loss_fns['preterm'](output_max, label)

            val_loss_avg += loss_avg.item() / len(ValLoader)
            val_loss_max += loss_max.item() / len(ValLoader) 
            
            for key in metrics_avg.keys():
                metrics_avg[key](output_avg, label.to(torch.int))
                metrics_max[key](output_max, label.to(torch.int))
    

    torch.save(model.state_dict(), save_path + '/weights/' + str(epoch).zfill(3) + '.pth')        

    logger_avg.log_metrics(metrics_avg, train_loss, val_loss_avg)
    logger_max.log_metrics(metrics_avg, train_loss, val_loss_max)

    logger_avg.plot_metrics()
    logger_max.plot_metrics()
   
        
