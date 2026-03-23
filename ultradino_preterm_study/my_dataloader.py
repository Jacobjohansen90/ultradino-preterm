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
import cv2


FUS13M_MEAN = 0.1842924807
FUS13M_STD = 0.2187705424

class PreTermDataset(Dataset):
    def __init__(self, data_path, cutoff=37, train=True):

        super().__init__()
        f = open(data_path)
        d = json.load(f)
        df = pd.DataFrame.from_dict(d)
        self.df = df.T
        self.img_size = [224,224]
        self.ga_cutoff = cutoff

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
        if np.isnan(data['pdx']):
            pdx = 0.01
        else:
            pdx = data['pdx']
        if np.isnan(data['pdy']):
            pdy = 0.01
        else:
            pdy = data['pdy']
        img = self.resample(img, pdx, pdy)
        
        img = self.transforms(image=img)['image']
        
        img_data = torch.Tensor([pdx, pdy])
        
        cpr_child = data['cpr_child']        
        
        ga_weeks = int(data['GA_days'])//7        
        
        label = ga_weeks <= self.ga_cutoff
                
        label = torch.Tensor([label*1.])
        
        return {'image': img, 'ps': img_data, 'label': label, 'cpr_child': cpr_child}
    
    def resample(img, pdx, pdy):
        w_max = 18.353558778762817
        h_max = 13.765169084072113
        new_h = img.shape[0]*pdx
        new_w = img.shape[1]*pdy
        ratio = img.shape[0]/img.shape[1]
        new_dim = max((int(ratio*224 * new_h / h_max)),224), max((int(224 * new_w / w_max)), 224)
        
        h_ref, w_ref = new_dim

        delta_h = 224 - h_ref # 133
        uneven_h = delta_h%2 # 1
        start_h = delta_h // 2 + uneven_h # 66 - 1 = 65
        end_h = 224 - delta_h // 2 # 224 - 66 = 158  158 * 65 = 93
        
        delta_w = 224 - w_ref
        uneven_w = delta_w%2
        start_w = delta_w // 2 + uneven_w
        end_w = 224 - delta_w // 2
        canvas = np.zeros((224,224,3)).astype('uint8')
        data_resampled = cv2.resize(img, (w_ref, h_ref), interpolation=cv2.INTER_CUBIC)
        canvas[start_h:end_h, start_w:end_w,:] = data_resampled
        return canvas

        
    
def collate_fn(batch):
    return {'image': torch.stack([x['image'] for x in batch]),
            'ps': torch.stack([x['ps'] for x in batch]),
            'label': torch.stack([x['label'] for x in batch]),
            'cpr_child': [x['cpr_child'] for x in batch]}