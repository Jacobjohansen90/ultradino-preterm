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
from datetime import datetime
from filelock import FileLock

from dataloader.dataloader import PreTermDataset, collate_fn
from utils.model_utils import model_from_conf
from utils.metric_loader import get_test_metrics

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")

#%%Main

def test_model(folder_path, test_data_path, move=True, batch_size=128):
    cfg = OmegaConf.load(folder_path + 'conf.yaml')
    
    test_df = pl.read_parquet(test_data_path)
    TestData = PreTermDataset(test_df, cfg, train=False)
    TestLoader = DataLoader(TestData,
                            batch_size,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False,
                            num_workers=cfg.data.workers,
                            collate_fn=collate_fn)

    model = model_from_conf(cfg)
    
    dirs = os.listdir(folder_path + 'weights/')
    dirs.sort()
    
    results = []
    thresholds = {'avg': pl.read_csv(folder_path + 'Avg_metrics.csv'),
                  'max': pl.read_csv(folder_path + 'Max_metrics.csv')}

    
    for i, weights in enumerate(dirs):
        model.load_state_dict(torch.load(folder_path + 'weights/' + weights, weights_only=True))
        model.eval()

        epoch_results = {}
        dfs = []

        with torch.no_grad():
            for data in tqdm(TestLoader):
                
                outputs = model(data['imgs'].to(cfg.device.type),
                                data['img_data'].to(cfg.device.type),
                                data['ehr_data'].to(cfg.device.type))
            
                dfs.append(pl.DataFrame({"cpr": data["ID"],
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
            t = thresholds[eval_type][i]['SensAtSpec_cutoff'].item()
            
            metrics = get_test_metrics(cfg, t)
    
            for metric in metrics.values():
                metric(preds[eval_type], labels)
    
            sens_spec, cutoff = metrics['SensAtSpec'].compute()
            sens = metrics["Sens"].compute().item()
            spec = metrics["Spec"].compute().item()
            
            epoch_results[f"SensAtSpec_{eval_type}"] = round(sens_spec.item(), 3)
            epoch_results[f"cutoff_{eval_type}"] = round(cutoff.item(), 3)
            epoch_results[f"Sens_{eval_type}"] = round(sens, 3)
            epoch_results[f"Spec_{eval_type}"] = round(spec, 3)
            epoch_results[f"cutoff_val_{eval_type}"] = round(t, 3)
        
        epoch_results['weights'] = folder_path.replace('Current', 'Tested') + 'weights/' + weights
        
        results.append(epoch_results)

    results_df = pl.DataFrame(results)
    results_df.write_csv(folder_path + "test_metrics.csv")
    
    results_df = results_df.with_columns(pl.max_horizontal("SensAtSpec_avg", "SensAtSpec_max").alias("SensAtSpec_best"))
    results_df = results_df.with_columns(pl.when(pl.col("SensAtSpec_avg") >= pl.col("SensAtSpec_max"))
                                         .then(pl.lit("avg")).otherwise(pl.lit("max")).alias("best_type"))
    
    top_5 = (results_df.sort("SensAtSpec_best", descending=True).head(5))
    top_5.write_csv(folder_path + "top_5.csv")
    
    best = (results_df.sort("SensAtSpec_best", descending=True).head(1))
    
    shutil.move(folder_path, folder_path.replace('Current', 'Tested'))
    
    save_path, name = folder_path.split('Current')
    name = name.replace('/','')
    
    sota_csv = os.path.join(save_path + f"SOTA/SOTA_{cfg.data.ga_cutoff_weeks}.csv")
    lock_file = sota_csv + ".lock"
    
    with FileLock(lock_file):
        dst_name = save_path + 'SOTA/weights/'
        if os.path.exists(sota_csv):
            sota_df = pl.read_csv(sota_csv)     
            if len(sota_df) < 5:
                shutil.copy(best['weights'][0], dst_name + name  + ".pth")
                shutil.copy(best['weights'][0].split('weights')[0] + 'conf.yaml' , dst_name + name  + ".yaml")
                best = best.with_columns(pl.lit(dst_name + name + '.pth').alias("weights")) 
                sota_df = pl.concat([sota_df, best])
    
            else:
                if best["SensAtSpec_best"][0] > sota_df["SensAtSpec_best"].min():
                    shutil.copy(best['weights'][0], dst_name + name  + ".pth")
                    shutil.copy(best['weights'][0].split('weights')[0] + 'conf.yaml', dst_name + name  + ".yaml")
                    best = best.with_columns(pl.lit(dst_name + name + '.pth').alias("weights")) 
                    sota_df = pl.concat([sota_df, best])
            
            sota_df = (sota_df.sort("SensAtSpec_best", descending=True).head(5))
            sota_df.write_csv(sota_csv)

            valid_weights = set(sota_df["weights"].to_list())        
            for file in os.listdir(dst_name):
                path = os.path.join(dst_name, file)
        
                if path not in valid_weights:
                    os.remove(path)
                    os.remove(path.replace('.pth', '.yaml'))
    
        else:
            os.makedirs(dst_name, exist_ok=False)
            shutil.copy(best['weights'][0], dst_name + name  + ".pth")
            shutil.copy(best['weights'][0].split('weights')[0] + 'conf.yaml', dst_name + name  + ".yaml")
            best = best.with_columns(pl.lit(dst_name + name + '.pth').alias("weights")) 
            best.write_csv(sota_csv)
    
        

    
    
    