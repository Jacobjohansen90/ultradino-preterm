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
import os
import polars as pl
import csv
import shutil

from dataloader.dataloader import PreTermDataset, collate_fn
from utils.model_utils import model_from_conf
from utils.metric_loader import get_test_metrics

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")

#%%Main

def test_model(folder_path, testdata_path, move=True):
    cfg = OmegaConf.load(folder_path + 'conf.yaml')
    test_df = pl.read_parquet(testdata_path)
    TestData = PreTermDataset(test_df, cfg, train=False)
    TestLoader = DataLoader(TestData,
                            1,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False,
                            num_workers=cfg.data.workers,
                            collate_fn=collate_fn)

    model = model_from_conf(cfg)
    
    f_all = open(folder_path + 'test_all.csv', 'w')
    wr_all = csv.writer(f_all)
    wr_all.writerow(['Sens@Spec_avg', 'Sens@Spec_max', 
                     'Sens_avg', 'Spec_avg',
                     'Sens_max', 'Spec_max'
                     'avg_cutoff', 'max_cutoff', 
                     'avg_cutoff_val', 'max_cutoff_val',
                     'weights'])
    
    df_avg = pl.read_csv(folder_path + 'Avg_metrics.csv')
    df_max = pl.read_csv(folder_path + 'Max_metrics.csv')
    
    dirs = os.listdir(folder_path + 'weights/')
    dirs.sort()
    
    for i, weights in enumerate(dirs):
        model.load_state_dict(torch.load(folder_path + 'weights/' + weights, weights_only=True))
        model.eval()
        
        t_avg = df_avg[i]['SensAtSpec_cutoff'].item()
        t_max = df_max[i]['SensAtSpec_cutoff'].item()

        metrics_avg = get_test_metrics(cfg, t_avg)
        metrics_max = get_test_metrics(cfg, t_max)

        
        with torch.no_grad():
            for data in iter(TestLoader):
                outputs = model(data['img'].to(cfg.device.type), 
                                data['img_data'].to(cfg.device.type), 
                                data['ehr_data'].to(cfg.device.type))
                
                output_avg = outputs['preterm'].mean().unsqueeze(0)
                output_max = outputs['preterm'].max().unsqueeze(0)
                label = data['labels']['preterm'][0].to(cfg.device.type)
                
                
                for key in metrics_avg.keys():
                    metrics_avg[key](output_avg, label.to(torch.int))
                    metrics_max[key](output_max, label.to(torch.int))
            
            SensSpec_avg = round(metrics_avg['SensAtSpec'].compute()[0].item(), 3)
            cutoff_avg = round(metrics_avg['SensAtSpec'].compute()[1].item(), 3)
            Sens_avg = round(metrics_avg['Sens'].compute().item(), 3)
            Spec_avg = round(metrics_avg['Spec'].compute().item(), 3)

            SensSpec_max = round(metrics_avg['SensAtSpec'].compute()[0].item(), 3)
            cutoff_max = round(metrics_avg['SensAtSpec'].compute()[1].item(), 3)
            Sens_max = round(metrics_avg['Sens'].compute().item(), 3)
            Spec_max = round(metrics_avg['Spec'].compute().item(), 3)
            
            wr_all.writerow([SensSpec_avg, SensSpec_max, Sens_avg, Spec_avg, Sens_max, Spec_max, 
                             cutoff_avg, cutoff_max, t_avg, t_max, weights])

    f_all.close()
    
    df = pl.read_csv(folder_path + 'test_all.csv')
        
    top_5 =  df.with_row_index("row_id").unpivot(index="row_id",
                                                 on=["Sens@Spec_avg", "Sens@Spec_max"],
                                                 variable_name="column",
                                                 value_name="value").top_k(5, by="value")
    
    with open(folder_path + 'test_top_5.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['Sens@Spec', 'type', 'Sens', 'Spec', 'cutoff', 'cutoff_val', 'weights'])

        for i in range(5):
            idx, col, _ = top_5.row(i)
            row = df[idx]
            t = col.split('_')[1]
            wr.writerow([row[col], t, row['Sens_'+t], row['Spec_'+t], row[t+'_cutoff'], row[t+'_cutoff_val'], row['weights']])
       
    if move:
        dst = folder_path.replace('Current', 'Tested')
        shutil.move(folder_path, dst)
    