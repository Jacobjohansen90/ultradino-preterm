#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 09:56:33 2026

@author: jacob
"""

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import json

from dataloader.dataloader import PreTermDataset, collate_fn, make_train_val_split
from utils.model_utils import model_from_conf

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")

cfg = OmegaConf.load("/projects/users/data/UCPH/DeepFetal/projects/preterm/Training_runs/Tested/2026-06-13 10:41:22/conf.yaml")
weights = "/projects/users/data/UCPH/DeepFetal/projects/preterm/Training_runs/Tested/2026-06-13 10:41:22/weights/017.pth"
save_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/misc/"


cfg.data.val_frac = 0
cfg.data.oversample = False



model = model_from_conf(cfg)
model.load_state_dict(torch.load(weights, weights_only=True))
model.eval()

for path in [cfg.data.path, cfg.data.test_path]:
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
    data = {}
    for i, data in enumerate(tqdm(Data)):
        idx = DataSet.groups[i][0]
        CPR = DataSet.df[idx]['CPR_CHILD'].item()
        data[CPR] = {}
        with torch.no_grad():

            outputs = model(data['img'].to(cfg.device.type), 
                            data['img_data'].to(cfg.device.type), 
                            data['ehr_data'].to(cfg.device.type))
            

            data[CPR]['pred'] = outputs['preterm'].to('cpu').tolist()
            data[CPR]['embedding'] = outputs['vision_features'].to('cpu').tolist()
            
    
    if path == cfg.data.path:
        with open(save_path + 'train.json', 'w') as f:
            json.dump(data, f)
    else:
        with open(save_path + 'test.json', 'w') as f:
            json.dump(data, f)


