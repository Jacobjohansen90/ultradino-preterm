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
import json 
import csv

cfg = OmegaConf.load("Vit_Small_Img_Resampled_B2M_cervical.yaml")

folds_path = "../weights/all_folds37w/All_Folds_Spacing_CL2026-03-13_13-21-52/"

data_path = '../../data/cervix_data_all.json'

metrics = get_metrics('cuda')

ValData = PreTermDataset(data_path, cutoff=37, train=False)

ValLoader = DataLoader(ValData,
                       128,
                       shuffle=False,
                       pin_memory=False,
                       drop_last=False,
                       num_workers=32,
                       collate_fn=collate_fn)



f = open("data.csv", 'w')
wr = csv.writer(f)
wr.writerow(['cpr', 'pred', 'label', 'fold'])

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
                preds = model(data['image'].to('cuda'), data['ps'].to('cuda'))
                pred = preds['preterm']
                cprs = data['cpr_child']
                labels = data['label']
                for j in range(len(cprs)):
                    wr.writerow([cprs[j], pred[j], labels[j], folder])
            

            
f.close()      




