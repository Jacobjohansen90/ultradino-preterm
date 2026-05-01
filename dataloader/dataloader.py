#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:33:56 2026

@author: jacob
"""

from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import albumentations as A
import json
import random

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
        
        self.imgs = np.abs(np.random.randn(scans, self.img_size[0], self.img_size[1]))
        self.pixel_spacings = torch.randn(scans, 2, dtype=torch.float32)
        self.ehr_data = torch.randint(14, 60, (scans,1,1), dtype=torch.float32)
        self.labels = torch.round(torch.rand(((500,1)), dtype=torch.float32))
    
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
        img = self.imgs[idx]
        img = img.astype(np.float32)
        img = self.transforms(image=img)['image']
        pixel_spacing = self.pixel_spacings[idx]
        
        ehr_data = self.ehr_data[idx]
        label = self.labels[idx]
        
        return {'img': img, 'img_data': pixel_spacing, 'ehr_data': ehr_data, 'label': label}
        
class PreTermDataset(Dataset):
    def __init__(self, data_dict, cfg, train):

        super().__init__()
        self.df = unpack_dict_to_DF(data_dict, 'imgs')
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
        data = self.df[int(idx)]
                
        ehr_data = []
        
        for key in self.ehr_data:
            ehr_data.append([float(data[key])])
        ehr_data = torch.Tensor(ehr_data)
        
        ga_weeks = int(data['GA'].item())//7        
        
        label = ga_weeks <= self.ga_cutoff
                
        label = torch.Tensor([label*1.])
        
        img = Image.open(self.prefix + data['file_path'].item())
        img = np.asarray(img)
        try:
            img = self.transforms(image=img)['image']
        except:
            img = torch.Tensor(np.zeros((1,224,224)))
            label = torch.Tensor([0])
        img_data = torch.Tensor([[0],[0]])
        # img_data = torch.Tensor([data['physical_delta_x'], data['physical_delta_y']])
        label.to(torch.int)
        
        return {'img': img, 'img_data': img_data, 'ehr_data': ehr_data, 'label': label}
    
def collate_fn(batch):
    return {'img': torch.stack([x['img'] for x in batch]),
            'img_data': torch.stack([x['img_data'] for x in batch]),
            'ehr_data': torch.stack([x['ehr_data'] for x in batch]),
            'label': torch.stack([x['label'] for x in batch])}


def make_train_val_split(cfg):
    with open(cfg.data.path) as file:
        d = json.load(file)
    keys = list(d.keys())
    random.shuffle(keys)
    
    split_idx = int(len(keys) * (1-cfg.data.val_frac))
    
    train_keys = keys[:split_idx]
    val_keys = keys[split_idx:]
    
    train_dict = {k: d[k] for k in train_keys}
    val_dict = {k: d[k] for k in val_keys}
    
    return train_dict, val_dict