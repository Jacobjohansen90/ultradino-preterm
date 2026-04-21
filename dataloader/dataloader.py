#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:33:56 2026

@author: jacob
"""

from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import json

from utils.utils import unpack_dict_to_DF

FUS13M_MEAN = 0.1842924807
FUS13M_STD = 0.2187705424

class DummySet(Dataset):
    def __init__(self, train, img_size=[224,224], scans=500):
        super().__init__()
        self.img_size = img_size
        self.scans = scans

        self.norm_mean = 0.1842924807
        self.norm_std = 0.2187705424

        self.setup_transforms(train)
        
    
    def __len__(self):
        return self.scans
    
    def setup_transforms(self, train):
        if train:
            self.transforms = A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
                                         A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                                         A.GaussNoise(std_range=(0.05, 0.2), p=0.5),
                                         A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), p=0.5),
                                         A.HorizontalFlip(p=0.5),                 
                                         A.Resize(height=self.img_size[0], width=self.img_size[1]),
                                         A.ToGray(p=1.0, num_output_channels=1),
                                         A.Normalize(mean=self.norm_mean, std=self.norm_std),
                                         A.ToTensorV2()])
        
        else:
            self.transforms = A.Compose([A.Resize(height=self.img_size[0], width=self.img_size[1]),
                                         A.ToGray(p=1.0, num_output_channels=1),
                                         A.Normalize(mean=self.norm_mean, std=self.norm_std),
                                         A.ToTensorV2()])

    
    def __getitem__(self, idx):
        img = np.abs(np.random.randn(self.img_size[0], self.img_size[1]))
        img = img.astype(np.float32)
        img = self.transforms(image=img)['image']
        pixel_spacing = torch.randn(2, dtype=torch.float32)
        
        ehr_data = torch.randint(14, 60, (1,1), dtype=torch.float32)
        
        label = torch.round(torch.rand((1,), dtype=torch.float32))
        
        return {'img': img, 'pixel_spacing': pixel_spacing, 'ehr_data': ehr_data, 'label': label}
        
class PreTermDataset(Dataset):
    def __init__(self, cfg, train):

        super().__init__()
        with open(cfg.data.path) as f:
            d = json.load(f)
        self.df = unpack_dict_to_DF(d, 'imgs')
        self.img_size = cfg.data.size
        self.ehr_data = cfg.data.ehr_data
        self.ga_cutoff = cfg.data.ga_cutoff_weeks
        self.prefix = cfg.data.prefix
        self.norm_mean = 0.1842924807
        self.norm_std = 0.2187705424        

        self.setup_transforms(train)
        
    def __len__(self):
        return len(self.df)
    
    def setup_transforms(self, train):
        if train:
            self.transforms = A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
                                         A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                                         A.GaussNoise(std_range=(0.05, 0.2), p=0.5),
                                         A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), p=0.5),
                                         A.HorizontalFlip(p=0.5),                 
                                         A.Resize(height=self.img_size[0], width=self.img_size[1]),
                                         A.ToGray(p=1.0, num_output_channels=1),
                                         A.Normalize(mean=self.norm_mean, std=self.norm_std),
                                         A.ToTensorV2()])
        
        else:
            self.transforms = A.Compose([A.Resize(height=self.img_size[0], width=self.img_size[1]),
                                         A.ToGray(p=1.0, num_output_channels=1),
                                         A.Normalize(mean=self.norm_mean, std=self.norm_std),
                                         A.ToTensorV2()])        
    
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        
        img = Image.open(self.prefix + data['file_path'])
        img = np.asarray(img)
        img = self.transforms(image=img)['image']
        
        img_data = torch.Tensor([data['physical_delta_x'], data['physical_delta_y']])
        
        ehr_data = []
        
        for key in self.ehr_data:
            if data[key] == '.':
                ehr_data.append([30.])
            else:
                ehr_data.append([float(data[key])])
        ehr_data = torch.Tensor(ehr_data)
        
        ga_weeks = int(data['GA_days'])//7        
        
        label = ga_weeks <= self.ga_cutoff
                
        label = torch.Tensor([label*1.])
        
        return {'img': img, 'img_data': img_data, 'ehr_data': ehr_data, 'label': label}
    
def collate_fn(batch):
    return {'img': torch.stack([x['img'] for x in batch]),
            'img_data': torch.stack([x['img_data'] for x in batch]),
            'ehr_data': torch.stack([x['ehr_data'] for x in batch]),
            'label': torch.stack([x['label'] for x in batch])}