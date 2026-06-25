#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:14:24 2026

@author: jacob
"""

from omegaconf import OmegaConf
import os
import polars as pl
import matplotlib.pyplot as plt

def setup(cfg):
    if cfg.info.name is None:
        raise Exception("Model experiment must be named")
    
    path = "../training_runs/Running/" + cfg.info.name + '/'
    os.makedirs(path + 'weights/', exist_ok=False)
    OmegaConf.save(cfg, path + 'conf.yaml')        
    return path

class Logger():
    def __init__(self, save_path):
        self.save_path = save_path
        self.metrics = pl.DataFrame()
    def log_metrics(self, metrics_dict, train_loss, val_loss):
        metrics = {}
        for eval_type in metrics_dict.keys():
            metrics[eval_type] = {}
            for key in metrics_dict[eval_type].keys():
                if key == 'SensAtSpec' or key == 'SpecAtSens':
                    metric, cutoff = metrics_dict[eval_type][key].compute().item()
                    metrics[eval_type][key] = round(metric, 3)
                    metrics[eval_type][key + '_cutoff'] = round(cutoff, 3)
                else:
                    metrics[eval_type][key] = round(metrics_dict[eval_type][key].compute().item(), 3)
        
        flat_df = {f"{metric}_{eval_type}": value for eval_type, metrics in metrics.items() for metric, value in metrics.items()}

        self.metrics = pl.concat((self.metrics, pl.DataFrame(flat_df))) 

        self.metrics.write_csv(self.save_path + 'metrics.csv')
        
        self.plot_metrics(metrics_dict.keys())
    
    def plot_metrics(self, keys):
        fig, axes = plt.subplots(1, len(keys), squeeze=False)
        axes = axes.ravel()
        for i, key in enumerate(keys):
            cols = [c for c in self.metrics.columns if key in c]
            for col in cols:
                axes[i].plot(self.metrics[col], label=col)
            axes[i].set_title(key)
            axes[i].legend(loc='upper left')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Metric Value')
            axes[i].set_ylim(0,1.05)

        plt.tight_layout()
        fig.savefig(self.save_path + 'metrics_plot.png', dpi=300)
        plt.close(fig)
                
              
    
        