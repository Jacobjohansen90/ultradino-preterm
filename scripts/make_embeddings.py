#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 09:56:33 2026

@author: jacob
"""

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from dataloader.dataloader import PreTermDataset, collate_fn, make_train_val_split
from utils.model_utils import model_from_conf

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")

cfg = OmegaConf.load("/projects/users/data/UCPH/DeepFetal/projects/preterm/training_runs/SOTA_37/conf.yaml")
weights = "/projects/users/data/UCPH/DeepFetal/projects/preterm/training_runs/SOTA_37/weights/019.pth"
save_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/misc/EHR/"


cfg.data.val_frac = 0
cfg.data.path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/"

path = cfg.data.path



model = model_from_conf(cfg)

for split in ['train.json', 'test.json']:
    cfg.data.path = path + split
    df, df_ = make_train_val_split(cfg, unique_column='CPR_MOTHER')
    if len(df_) > 0:
        raise Exception("Data in val split")
 

    DataSet = PreTermDataset(df, cfg, train=False)

    Data = DataLoader(DataSet,
                      1,
                      shuffle=False,
                      pin_memory=False,
                      drop_last=False,
                      num_workers=8,
                      collate_fn=collate_fn)

    with open(cfg.data.path) as f:
        data_dict = json.load(f) 

    for i, data in enumerate(tqdm(Data)):
        CPR = DataSet.df[i]['CPR_CHILD'].item()
        fp = DataSet.df[i]['file_path'].item()
        
        outputs = model(data['img'].to(cfg.device.type), 
                        data['img_data'].to(cfg.device.type), 
                        data['ehr_data'].to(cfg.device.type))
        
        found = False
        for i in range(len(data_dict[CPR]['imgs'])):
            if data_dict[CPR]['imgs'][i]['file_path'] == fp:
                found = True
                data_dict[CPR]['imgs'][i]['pred'] = outputs['preterm']
                data_dict[CPR]['imgs'][i]['logits'] = outputs['preterm_logits']
        
        if not found:
            print(f"{fp} was not found in child {CPR}") 
    
    
    with open(save_path + split) as f:
        json.dump(data_dict, f)


