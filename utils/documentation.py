#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:14:24 2026

@author: jacob
"""

from omegaconf import OmegaConf
from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt

def setup(conf):
    
    timestamp =  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = "../training_runs/" + timestamp + '/'
    os.makedirs(path, exist_ok=False)
    os.makedirs(path + 'weights/', exist_ok=False)
    OmegaConf.save(conf, path + 'conf.yaml')
        
    return path

class Logger():
    def __init__(self, save_path):
        self.save_path = save_path
        self.first_log = True
        self.metrics = []
        self.headers = ['train_loss', 'val_loss']

    def log_metrics(self, metrics_dict, train_loss, val_loss):
        if self.first_log:
            f = open(self.save_path + 'metrics.csv', 'w')
            f.close()
            headers = ['train_loss', 'val_loss']
            for key in metrics_dict.keys():
                headers.append(key)
                self.headers.append(key)

            with open(self.save_path + 'metrics.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
            self.first_log = False

        metrics = [round(train_loss, 3), round(val_loss, 3)]
        
        for key in metrics_dict.keys():
            if key == 'SensAtSpec':
                metrics.append(round(metrics_dict[key].compute()[0].item(), 3))
            else:
                metrics.append(round(metrics_dict[key].compute().item(), 3))
            
        with open(self.save_path + 'metrics.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(metrics)
        self.metrics.append(metrics)    
    
    def plot_metrics(self):
        plt.plot(self.metrics, label=self.headers)
        plt.legend(loc='upper left')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.savefig(self.save_path + 'Metrics_plot.png')
        plt.close()
                
              
    
        