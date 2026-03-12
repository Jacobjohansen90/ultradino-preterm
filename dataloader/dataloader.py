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
        
        ga = torch.randint(25, 50, (1,), dtype=torch.float32)
        
        return {'img': img, 'pixel_spacing': pixel_spacing, 'ehr_data': ehr_data, 'ga': ga}
        
class PreTermDataset(Dataset):
    def __init__(self,
                 csv_path,
                 train,
                 resize=[224,224],
                 ehr_data=['mothers_age']):

        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.resize = resize
        self.ehr_data = ehr_data

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
        
        pixel_spacing = torch.Tensor(data['pixel_spacing'])
        
        ehr_data = []
        
        for key in self.ehr_data:
            ehr_data.append(data[key])
        
        ehr_data = torch.Tensor(ehr_data)
        
        ga = torch.Tensor(data['ga'])        
        
        return {'img': img, 'pixel_spacing': pixel_spacing, 'ehr_data': ehr_data, 'ga': ga}