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

class DummySet(Dataset):
    def __init__(self, size=[224,224], scans=500):
        super().__init__()
        self.size = size
        self.scans = scans
    
    def __len__(self):
        return self.scans
    
    def __getitem__(self, idx):
        img = torch.randn(self.size)
        pixel_spacing = torch.randn(2)
        
        ehr_data = torch.randint(14, 60, (1,))
        
        ga = torch.randint(25, 50, (1,))
        
        return {'img': img, 'pixel_spacing': pixel_spacing, 'ehr_data': ehr_data, 'ga': ga}
        
class PreTermDataset(Dataset):
    def __init__(self,
                 csv_path,
                 resize=[224,224],
                 ehr_data=['mothers_age']):

        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.resize = resize
        self.ehr_data = ehr_data
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        
        img = Image.open(data['img_path'])
        img = img.resize(self.resize)
        img = np.asarray(img)
        img = torch.from_numpy(img)
        
        pixel_spacing = torch.Tensor(data['pixel_spacing'])
        
        ehr_data = []
        
        for key in self.ehr_data:
            ehr_data.append(data[key])
        
        ehr_data = torch.Tensor(ehr_data)
        
        ga = torch.Tensor(data['ga'])        
        
        return {'img': img, 'pixel_spacing': pixel_spacing, 'ehr_data': ehr_data, 'ga': ga}
    
    