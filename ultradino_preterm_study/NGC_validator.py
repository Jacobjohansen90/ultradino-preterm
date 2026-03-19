#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:47:00 2026

@author: jacob
"""

from model.model import PretermModel, PretermFinetuning
from omegaconf import OmegaConf
import os
from my_dataloader import PreTermDataset, collate_fn
from tqdm import tqdm
import torch
from torch.utils.data import  DataLoader
from metric_loader import get_metrics
import csv

cfg = OmegaConf.load("Vit_Small_Img_Resampled_B2M_cervical.yaml")

folds_path = "/home/jacob/Desktop/NAS/Work/all_folds/All_Folds_Spacing_CL2026-03-13_13-21-52/"

data_path = ''

metrics = get_metrics('cuda')

ValData = PreTermDataset(data_path, train=False)

ValLoader = DataLoader(ValData,
                       64,
                       shuffle=False,
                       pin_memory=False,
                       drop_last=False,
                       num_workers=32,
                       collate_fn=collate_fn)


f = open('results.csv', 'w')
writer = csv.writer(f)

headers = ['Fold'] + list(metrics.keys())


for folder in os.listdir(folds_path):
    if 'fold' in folder:
        weights = os.listdir(folds_path + folder + '/checkpoints/')[0]
        weight_path = folds_path + folder + '/checkpoints/' + weights
        
        model = PretermModel.from_conf(cfg.model)

        model = PretermFinetuning.load_from_checkpoint(weight_path,
                                                       model=model, 
                                                       map_location='cuda')
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(ValLoader)):
                preds = model(data['image'].to('cuda'), data['pxs'].to('cuda'))
                labels = data['label'].to('cuda')
                for key in metrics.keys():
                    metrics[key](preds, labels)
            
            report = [folder]
            
            for key in metrics.keys():
                report.append(round(metrics[key].compute().item(), 3))

            writer.writerow(report)
            
        




