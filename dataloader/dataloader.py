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
import polars as pl

FUS13M_MEAN = 0.1842924807
FUS13M_STD = 0.2187705424

class DummySet(Dataset):
    def __init__(self, train, img_size=[224,224], scans=500):
        super().__init__()
        self.img_size = img_size
        self.scans = scans

        self.norm_mean = FUS13M_MEAN
        self.norm_std = FUS13M_STD

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
    def __init__(self, df, cfg, train, ID='CPR_CHILD'):

        super().__init__()
        self.img_size = cfg.data.img_size
        self.ehr_vars = cfg.data.ehr_data
        self.img_data_vars = cfg.data.img_data
        self.norm_mean = 0.1842924807
        self.norm_std = 0.2187705424       
        self.train = train
        self.setup_transforms()
        self.ID_var = ID
        self.df = df
        self.remove_on_GA_vars = []
        for var, cond in cfg.dataset.items():
            if cond == 'remove_on_GA':
                self.remove_on_GA_vars.append(var)
        
    
    def setup_transforms(self):
        if self.train:
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
        return self.getitem(idx)
        
    def population_count(self, ga_cutoff):
        not_preterm = self.df.filter(pl.col("GA")//7 >= ga_cutoff)["CPR_CHILD"].n_unique()
        preterm = self.df.filter((pl.col("GA")//7 < ga_cutoff) & (pl.all_horizontal(~pl.col(self.remove_on_GA_vars))))["CPR_CHILD"].n_unique()
        population = self.df["CPR_CHILD"].n_unique()
   
        return preterm, not_preterm, population
    
    def __len__(self):
        return len(self.df)


    def getitem(self, idx):
                
        #Get data as named dict
        data = self.df.row(idx, named=True)

        #Prepare EHR data
        ehr_data = []
        for key in self.ehr_vars:
            ehr_data.append(float(data.get(key)) or 0.0)
        ehr_data = torch.tensor(ehr_data)
        ehr_data = ehr_data.unsqueeze(0)

        
        #Prepare labels
        GA_weeks = data.get('GA')//7
        GA_weeks = torch.tensor([float(GA_weeks)])

        remove_on_GA = torch.tensor([0], dtype=torch.float32)
        for var in self.remove_on_GA_vars:
            if data.get(var):
                remove_on_GA = torch.tensor([1], dtype=torch.float32)

                
        #Prepare Image       
        img = Image.open(data.get('no_ocr_preprocessed_file_path'))
        img = np.asarray(img)
        img = self.transforms(image=img)['image']

        #Prepare image metadata
        img_data = []
        for key in self.img_data_vars:
            img_data.append(data.get(key) or 0.0)
        img_data = torch.tensor(img_data)
        img_data = img_data.unsqueeze(0)
                
        #Get patient identifier
        ID = data.get(self.ID_var)

        return {'img': img, 'img_data': img_data, 'ehr_data': ehr_data, 'GA_weeks': GA_weeks, 'ID': ID, 'remove_on_GA': remove_on_GA}


def collate_fn(batch):
    imgs = torch.stack([sample['img'] for sample in batch])
    img_data = torch.stack([sample['img_data'] for sample in batch])
    ehr_data = torch.stack([sample['ehr_data'] for sample in batch])
    GA_weeks = torch.stack([sample['GA_weeks'] for sample in batch])
    IDs = [sample['ID'] for sample in batch]
    remove_on_GA = torch.stack([sample['remove_on_GA'] for sample in batch])

    sample =  {"imgs": imgs,
               "img_data": img_data,
               "ehr_data": ehr_data,
               "GA_weeks": GA_weeks,
               "IDs": IDs,
               "remove_on_GA": remove_on_GA}

    return sample
   

def make_data_split(cfg, data_path, unique_column='CPR_MOTHER', training=True):
    df = pl.read_parquet(data_path)
    
    for col, cond in cfg.dataset.items():
        if cond == 'remove':
            df = df.filter(~pl.col(col))
    
    if training:
        unique_keys = df.select(unique_column).unique()
    
        rng = np.random.default_rng()
        keys = unique_keys.to_series().to_list()
        rng.shuffle(keys)
    
        split_idx = int(len(keys) * (1 - cfg.data.val_frac))
        train_keys = keys[:split_idx]
        val_keys = keys[split_idx:]
    
        train_df = df.filter(pl.col(unique_column).is_in(train_keys))
        val_df = df.filter(pl.col(unique_column).is_in(val_keys))
        
        if cfg.data.oversample_ratio != 0:
            df_1 = train_df.filter(pl.col('GA')//7 < max(cfg.tasks.preterm.cutoffs))
            df_0 = train_df.filter(pl.col('GA')//7 >= max(cfg.tasks.preterm.cutoffs))
            n1 = df_1.height
            n0 = df_0.height
            if n1 > n0:
                df_0 = df_0.sample(n=n1*cfg.data.oversample_ratio, with_replacement=True)
            else:
                df_1 = df_1.sample(n=n0*cfg.data.oversample_ratio, with_replacement=True)
            train_df = pl.concat([df_1, df_0])
            train_df = train_df.sample(fraction=1.0, shuffle=True)
        if (train_df[unique_column].is_in(val_df[unique_column].implode())).any():
            raise Exception(f"Traindata and Validation data overlap on column {unique_column}")
        
        return train_df, val_df
    
    else:
        return df