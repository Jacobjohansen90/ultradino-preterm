#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:03:57 2026

@author: jacob
"""

import torchmetrics.classification as tm
import polars as pl
import torch
from pathlib import Path
import matplotlib.pyplot as plt

class Metrics():
    def __init__(self, cfg, save_path):
        cutoffs = cfg.tasks.preterm.cutoffs
        metric = tm.SensitivityAtSpecificity(min_specificity=0.85, task='binary').to(cfg.device.type)

        dfs = {str(c): [] for c in cutoffs}
        metrics = {agg: {str(c): {'SensAtSpec': [],
                                  'SensAtSpec_cutoff': []} 
                         for c in cutoffs} for agg in ("avg", "max")}
        
        self.cutoffs = cutoffs
        self.metric = metric
        self.metrics = metrics
        self.dfs = dfs
        self.save_path = save_path
        
    def update(self, outputs, data):
        print(outputs)
        print(data)
        for cutoff in self.cutoffs:
            self.dfs[str(cutoff)].append(pl.DataFrame({'cpr': data['IDs'],
                                                       'preds': outputs['preterm'][str(cutoff)]['preds'].cpu().numpy(),
                                                       'label': (data['GA_weeks'] < float(cutoff)).cpu().numpy()}))
    
    def plot_metrics(self):
        for agg in ["avg", "max"]:
            fig, ax = plt.subplots(figsize=(8, 4))
            
            for cutoff in self.cutoffs:
                ax.plot(self.metrics[agg][str(cutoff)]['SensAtSpec'],
                        label=f"{cutoff} weeks")
        
            ax.set_title(agg.capitalize())
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Sensitivity @ 85% Specificity")
            ax.set_ylim(0, 1.05)
            ax.legend(loc="upper left")
    
            plt.tight_layout()
            fig.savefig(Path(self.save_path) / f"{agg}_metrics.png", dpi=300)
            plt.close(fig)
        
    def log_metrics(self, train_loss, val_loss):

        row = {'train_loss': round(train_loss, 5), 'val_loss': round(val_loss, 5)}

        for cutoff in self.cutoffs:
            df = pl.concat(self.dfs[str(cutoff)])
            print(df)
            patient_df = (df.group_by("cpr").agg([pl.col('preds').mean().alias('avg'),
                                                  pl.col('preds').max().alias('max'),
                                                  pl.col('label').first().alias('label')]))
            print('PATIENT_DF')
            print(patient_df)
            labels = torch.tensor(patient_df['label'].to_numpy(), dtype=torch.int32)
            
            for agg in ['avg', 'max']:
                preds = torch.tensor(patient_df[agg].to_numpy(), dtype=torch.float32)
                self.metric.reset()          
                result = self.metric(preds, labels)
                sens_spec, sens_spec_cutoff = self.metric(preds, labels)
                
                self.metrics[agg][str(cutoff)]['SensAtSpec'].append(sens_spec.item())
                self.metrics[agg][str(cutoff)]['SensAtSpec_cutoff'].append(sens_spec_cutoff.item())

                for name, values in self.metrics[agg][str(cutoff)].items():
                    row[f"{name}_{cutoff}_{agg}"] = values[-1]
                    
        metrics_df = pl.DataFrame([row])

        path = Path(self.save_path) / 'metrics.csv'
        
        if path.exists():
            existing = pl.read_csv(path)
            pl.concat([existing, metrics_df], how="vertical").write_csv(path)
        else:
            metrics_df.write_csv(path)
        
        self.plot_metrics()
        self.dfs = {str(c): [] for c in self.cutoffs}

        
def get_metrics(cfg, t=0.5):
    metrics = {'Recall': tm.Recall(task='binary', threshold=t).to(cfg.device.type),
               'Specificity': tm.Specificity(task='binary', threshold=t).to(cfg.device.type),
               'SensAtSpec': tm.SensitivityAtSpecificity(min_specificity=0.85, task='binary').to(cfg.device.type)}
    
    return metrics
    
