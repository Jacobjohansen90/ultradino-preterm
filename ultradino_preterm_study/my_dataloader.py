#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:05:28 2026

@author: jacob
"""

from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import json


FUS13M_MEAN = 0.1842924807
FUS13M_STD = 0.2187705424

class PreTermDataset(Dataset):
    def __init__(self, data_path, train):

        super().__init__()
        f = open(data_path)
        d = json.load(f)
        df = pd.DataFrame.from_dict(d)
        self.df = df.T
        self.img_size = [224,224]
        self.ga_cutoff = 37

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
        
        img = Image.open(data['img_path'])
        img = np.asarray(img)
        img = self.transforms(image=img)['image']
        
        img_data = torch.Tensor([data['pdx'], data['pdy']])
        
        
        ga_weeks = int(data['GA_days'])//7        
        
        label = ga_weeks <= self.ga_cutoff
                
        label = torch.Tensor([label*1.])
        
        return {'image': img, 'ps': img_data, 'label': label}
    
def collate_fn(batch):
    return {'image': torch.stack([x['image'] for x in batch]),
            'ps': torch.stack([x['ps'] for x in batch]),
            'label': torch.stack([x['label'] for x in batch])}