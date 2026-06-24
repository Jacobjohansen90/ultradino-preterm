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
import polars as pl

from dataloader.dataloader import PreTermDataset, collate_fn, make_train_val_split
from utils.model_utils import model_from_conf

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")

path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Training_runs/Tested/2026-06-13 10:41:22/'
save_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/misc/"
cfg = OmegaConf.load(path + "conf.yaml")

result_df = pl.read_csv(path + 'test_all.csv')

row = (result_df.with_columns(max_val=pl.max_horizontal("Sens@Spec_avg", "Sens@Spec_max")).sort("max_val", descending=True).head(1))

weights = path + 'weights/' + row['weights']

print(weights)

cfg.data.val_frac = 0
cfg.data.oversample = False

model = model_from_conf(cfg)
model.load_state_dict(torch.load(weights, weights_only=True))
model.eval()

for parquet in [cfg.data.path, cfg.data.test_path]:

    df = make_train_val_split(cfg, unique_column='CPR_MOTHER', is_test=True)

    DataSet = PreTermDataset(df, cfg, train=False, ID='no_ocr_preprocessed_file_path')

    Data = DataLoader(DataSet,
                      128,
                      shuffle=False,
                      pin_memory=False,
                      drop_last=False,
                      num_workers=64,
                      collate_fn=collate_fn)
    
    
    dfs = []
    embeddings = {}
    
    for data in tqdm(Data):
            
        outputs = model(data['imgs'].to(cfg.device.type),
                        data['img_data'].to(cfg.device.type),
                        data['ehr_data'].to(cfg.device.type))
    
        dfs.append(pl.DataFrame({"img": data["ID"],
                                 "preterm_pred": outputs["preterm"].flatten().cpu().numpy(),
                                 "preterm_label": data["labels"]["preterm"].flatten().cpu().numpy()}))
        
        embeddings.update({id_: emb for id_, emb in zip(data["ID"], outputs['vision_features'].to('cpu').tolist())})

    pred_df = pl.concat(dfs)
    
    df_final = df.join(pred_df, left_on='no_ocr_preprocessed_file_path', right_on='imgs', how='left')
    
    name = '_'.join(parquet.split('/')[-2:])
    df_final.write_parquet(save_path + name)

    with open(save_path + name.replace('.parquet', '.json')) as f:
        json.dump(embeddings, f)

