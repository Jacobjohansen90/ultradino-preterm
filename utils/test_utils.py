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
import shutil
from tqdm import tqdm
from filelock import FileLock

from dataloader.dataloader import PreTermDataset, collate_fn, make_data_split
from utils.model_utils import model_from_conf
from utils.metric_loader import get_metrics

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")

#%%Main

def test_model(folder_path, move=True, batch_size=128):
    cfg = OmegaConf.load(folder_path + 'conf.yaml')
    
    cfg.dataset.progesterone = 'ignore'
    df = make_data_split(cfg, cfg.data.test_path, training=False)
    TestDataProg = PreTermDataset(df, cfg, train=False)
    TestLoaderProg = DataLoader(TestDataProg,
                                batch_size,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False,
                                num_workers=cfg.data.workers,
                                collate_fn=collate_fn)

    
    cfg.dataset.progesterone = 'remove'
    df = make_data_split(cfg, cfg.data.test_path, training=False)
    TestDataNoProg = PreTermDataset(df, cfg, train=False)
    TestLoaderNoProg = DataLoader(TestDataNoProg,
                                  batch_size,
                                  shuffle=False,
                                  pin_memory=False,
                                  drop_last=False,
                                  num_workers=cfg.data.workers,
                                  collate_fn=collate_fn)
    
    model = model_from_conf(cfg)
    
    dirs = os.listdir(folder_path + 'weights/')
    dirs.sort()
    metrics_df =  pl.read_csv(folder_path + 'metrics.csv')
    thresholds = {'avg': metrics_df['SensAtSpec_cutoff_avg'],
                  'max': metrics_df['SensAtSpec_cutoff_max']}

    
    preterm_all, population_all = TestDataProg.population_count()
    preterm_np, population_np = TestDataNoProg.population_count() 
    
    best_epoch = {'all': {'population': population_all,
                          'preterm': preterm_all,
                          'SensAtSpec': 0.},
                  'np': {'population': population_np,
                         'preterm': preterm_np,
                         'SensAtSpec': 0.}}


    for i, weights in enumerate(dirs):
        weight_path = folder_path + 'weights/' + weights
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        model.eval()
        
        
        with torch.no_grad():
            for loader, population in [[TestLoaderProg, 'all'], [TestLoaderNoProg, 'np']]:
                dfs = []
                for data in tqdm(loader):
                    
                    outputs = model(data['imgs'].to(cfg.device.type),
                                    data['img_data'].to(cfg.device.type),
                                    data['ehr_data'].to(cfg.device.type))
                
                    dfs.append(pl.DataFrame({"cpr": data["IDs"],
                                             "pred": outputs["preterm"].flatten().cpu().numpy(),
                                             "label": data["labels"]["preterm"].flatten().cpu().numpy()}))
    
                pred_df = pl.concat(dfs)
    
                patient_df = (pred_df.group_by("cpr").agg([pl.col("pred").mean().alias("pred_avg"),
                                                           pl.col("pred").max().alias("pred_max"),
                                                           pl.col("label").first().alias("label")]))
            
                preds = {'avg': torch.tensor(patient_df["pred_avg"].to_numpy(), dtype=torch.float32),
                         'max': torch.tensor(patient_df["pred_max"].to_numpy(), dtype=torch.float32)}
        
                labels = torch.tensor(patient_df["label"].to_numpy(), dtype=torch.int)
                
                for eval_type in ['avg', 'max']: 
        
                    t = thresholds[eval_type].item(i)
                    metrics = get_metrics(cfg, t)
            
                    for metric in metrics.values():
                        metric(preds[eval_type], labels)
            
                    sens_spec, cutoff = metrics['SensAtSpec'].compute()
                    
                    if sens_spec.item() > best_epoch[population]['SensAtSpec']:
                        best_epoch[population]['Epoch'] = i
                        best_epoch[population]['SensAtSpec'] = sens_spec.item()
                        best_epoch[population]['Type'] = eval_type
                        best_epoch[population]['Sensitivity'] = metrics["Recall"].compute().item()
                        best_epoch[population]['Specificity'] = metrics["Specificity"].compute().item()
                        best_epoch[population]['SensAtSpec_cutoff'] = cutoff.item()
                        best_epoch[population]['Val_Cutoff'] = t
                        best_epoch[population]['weights'] = weight_path.replace('Running', 'Evaluated')

    with open(folder_path + 'test_results.txt', 'w') as f:
        f.write('--All patients--\n')
        for key in best_epoch['all'].keys():
            f.write(f"\t {key} : {best_epoch['all'][key]}\n")
        f.write('\n')
        f.write('--No Progesterone patients--\n')
        for key in best_epoch['np'].keys():
            f.write(f"\t {key} : {best_epoch['np'][key]}\n")
    
    sota_path = folder_path.split('Running')[0] + 'SOTA/SOTA.csv'
    
    lock_file = sota_path + ".lock"
    
    with FileLock(lock_file):
    
        if os.path.exists(sota_path):
            df = pl.read_csv(sota_path)
                
        else:
            df = pl.DataFrame({"SensAtSpec": pl.Series([], dtype=pl.Float64),
                               "population": pl.Series([], dtype=pl.String),
                               "GA": pl.Series([], dtype=pl.Int64),
                               "Weight_path": pl.Series([], dtype=pl.String)})
    
        for population in ['all', 'np']:
            cond = (pl.col("GA") == cfg.data.ga_cutoff_weeks) & (pl.col("population") == population)
            existing = df.filter(cond)
            value = existing["SensAtSpec"].item() if existing.height == 1 else 0
        
            if value < best_epoch[population]['SensAtSpec']:
                result = pl.DataFrame({"SensAtSpec": [best_epoch[population]['SensAtSpec']],
                                       "population": [population],
                                       "GA": [cfg.data.ga_cutoff_weeks],
                                       "Weight_path": [best_epoch[population]['weights']]})
                
                if existing.height == 0:
                    df = pl.concat([df, result], how="vertical")
                else:
                    df = df.with_columns([pl.when(cond).then(result[col][0]).otherwise(pl.col(col)).alias(col) for col in df.columns])

                os.makedirs(folder_path.split('Running')[0] + f"SOTA/{cfg.data.ga_cutoff_weeks}",
                            exist_ok=True)
                
                shutil.copytree(folder_path, 
                                folder_path.split('Running')[0] + f"SOTA/{cfg.data.ga_cutoff_weeks}/{population}/",
                                dirs_exist_ok=True)

        df = df.sort(["GA", "population"])
        df.write_csv(sota_path)
                

    if move:
        shutil.move(folder_path, folder_path.replace('Running', 'Evaluated'))
    
  
        

    
    
    