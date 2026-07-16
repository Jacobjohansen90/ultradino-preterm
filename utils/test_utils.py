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
from utils.metrics import get_metrics

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

    cutoffs = cfg.tasks.preterm.cutoffs
    
    metrics_df =  pl.read_csv(folder_path + 'metrics.csv')
    best_epoch = {}
    thresholds = {}
    best_preds = {c: None for c in cutoffs}
    
    for cutoff in cutoffs:
        pt_all, not_pt_all, pop_all = TestDataProg.population_count(cutoff)
        pt_no_prog, not_pt_no_prog, pop_no_prog = TestDataNoProg.population_count(cutoff)
    
        best_epoch[str(cutoff)] = {'all': {'population': pop_all,
                                           'preterm': pt_all,
                                           'not_preterm': not_pt_all,
                                           'SensAtSpec': 0.},
                                   'no_prog': {'population': pop_no_prog,
                                               'preterm': pt_no_prog,
                                               'not_preterm': not_pt_no_prog,
                                               'SensAtSpec': 0.}}
        
        thresholds[str(cutoff)] = {'avg': metrics_df[f"SensAtSpec_cutoff_{cutoff}_avg"],
                                   'max': metrics_df[f"SensAtSpec_cutoff_{cutoff}_max"]}
        
    for i, weights in enumerate(dirs):
        weight_path = folder_path + 'weights/' + weights
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        model.eval()
        
        with torch.no_grad():
            for loader, population in [[TestLoaderProg, 'all'], [TestLoaderNoProg, 'no_prog']]:
                dfs = {str(c): [] for c in cutoffs}
                for data in tqdm(loader):
                    outputs, _ = model(data['imgs'].to(cfg.device.type),
                                       data['img_data'].to(cfg.device.type),
                                       data['ehr_data'].to(cfg.device.type))
                    for cutoff in cutoffs:
                        dfs[str(cutoff)].append(pl.DataFrame({'cpr': data['IDs'],
                                                              'preds': outputs['preterm'][str(cutoff)]['preds'].flatten().cpu().numpy(),
                                                              'label': (data['GA_weeks'] < float(cutoff)).flatten().cpu().numpy()}))
                
                for cutoff in cutoffs:
                    pred_df = pl.concat(dfs[str(cutoff)])    
                    patient_df = (pred_df.group_by("cpr").agg([pl.col('preds').mean().alias('pred_avg'),
                                                               pl.col('preds').max().alias('pred_max'),
                                                               pl.col('label').first().alias('label')]))
            
                    preds = {'avg': torch.tensor(patient_df['pred_avg'].to_numpy(), dtype=torch.float32),
                             'max': torch.tensor(patient_df['pred_max'].to_numpy(), dtype=torch.float32)}
        
                    labels = torch.tensor(patient_df["label"].to_numpy(), dtype=torch.int32)
                
                    for eval_type in ['avg', 'max']: 
                        t = thresholds[str(cutoff)][eval_type].item(i)
                        metrics = get_metrics(cfg, t)
                
                        for metric in metrics.values():
                            metric(preds[eval_type], labels)
                
                        sens_spec, sens_spec_cutoff = metrics['SensAtSpec'].compute()
                        
                        best = best_epoch[str(cutoff)][population]
                        
                        if sens_spec.item() > best['SensAtSpec']:
                            best['Epoch'] = i
                            best['SensAtSpec'] = sens_spec.item()
                            best['Type'] = eval_type
                            best['Sensitivity'] = metrics['Recall'].compute().item()
                            best['Specificity'] = metrics['Specificity'].compute().item()
                            best['SensAtSpec_cutoff'] = sens_spec_cutoff.item()
                            best['Val_Cutoff'] = t
                            best['weights'] = weight_path.replace('Running', 'Evaluated')
                            best_preds[str(cutoff)] = patient_df[['cpr', f"pred_{eval_type}", 'label']]
                            
    
    os.makedirs(folder_path + 'preds/', exist_ok=True)
    for cutoff in cutoffs:
        best_preds[str(cutoff)].write_csv(folder_path + f"preds/GA_{cutoff}.csv")
        
    with open(folder_path + 'test_results.txt', 'w') as f:
        for cutoff in cutoffs:
            f.write(f"--GA {str(cutoff)}--\n\n")
            f.write('--All patients--\n')
            for key, value in best_epoch[str(cutoff)]['all'].items():
                if isinstance(value, float):
                    value = round(value, 3)
                f.write(f"\t {key} : {value}\n")
            f.write('\n')
            f.write('--No Progesterone patients--\n')
            for key, value in best_epoch[str(cutoff)]['no_prog'].items():
                if isinstance(value, float):
                    value = round(value, 3)
                f.write(f"\t {key} : {value}\n")
    
    
    sota_path = folder_path.split('Running')[0] + 'SOTA.csv'
    
    lock_file = sota_path + ".lock"
    with FileLock(lock_file):
        if os.path.exists(sota_path):
            df = pl.read_csv(sota_path)
        else:
            df = pl.DataFrame({"SensAtSpec": pl.Series([], dtype=pl.Float64),
                               "population": pl.Series([], dtype=pl.String),
                               "GA": pl.Series([], dtype=pl.Int64),
                               "Weight_path": pl.Series([], dtype=pl.String)})
            
    
        for cutoff in cutoffs:
            for population in ['all', 'no_prog']:
                best = best_epoch[str(cutoff)][population]
                
                cond = (pl.col("GA") == cutoff) & (pl.col("population") == population)
                existing = df.filter(cond)
                current_best = (existing["SensAtSpec"].item() if existing.height == 1 else 0.0)
                
                if best['SensAtSpec'] > current_best:
                    result = pl.DataFrame({"SensAtSpec": [best['SensAtSpec']],
                                           "population": [population],
                                           "GA": [cutoff],
                                           "Weight_path": [best['weights']]})
                
                    df = (pl.concat([df.filter(~cond), result]).sort(["GA", "population"]))

        df.write_csv(sota_path)
                

    if move:
        shutil.move(folder_path, folder_path.replace('Running', 'Evaluated'))
    
  
        

    
    
    