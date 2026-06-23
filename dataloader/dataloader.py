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
    def __init__(self, df, cfg, train):

        super().__init__()
        self.img_size = cfg.data.img_size
        self.ehr_vars = cfg.data.ehr_data
        self.img_data_vars = cfg.data.img_data
        self.ga_cutoff = cfg.data.ga_cutoff_weeks
        self.prefix = cfg.data.prefix
        self.norm_mean = 0.1842924807
        self.norm_std = 0.2187705424       
        self.train = train
        self.relabel = []
        
        if cfg.labels.label_smoothing:
            self.label_smoothing_param = cfg.labels.label_smoothing_param
        else:
            self.label_smoothing_param = None

        df = df.with_columns(pl.lit(False).alias('relabel'))

        for col, cond in cfg.dataset.items():
            if cond == 'ignore':
                continue
            elif cond == 'remove':
                df = df.filter(~pl.col(col))
            elif cond == 'remove_on_GA':
                df = df.filter(~(pl.col(col) & (pl.col('GA') // 7 < cfg.data.ga_cutoff_weeks)))
            elif cond == 'label':
                df = df.with_columns((pl.col('relabel') | pl.col(col)).alias('relabel'))            

        self.df = df

        self.groups = (df.with_row_index().group_by("CPR_CHILD", maintain_order=True)
                       .agg(pl.col("index"))["index"].to_list())
        
        self.setup_transforms()
        
    def __len__(self):
        if self.train:
            return len(self.df)
        else:
            return len(self.groups)
    
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
    
    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))
        
    def __getitem__(self, idx):
        return self.getitem(idx)
        
    def getitem(self, idx):
                
        #Get data as named dict
        data = self.df.row(idx, named=True)

        #Prepare EHR data
        ehr_data = []
        for key in self.ehr_vars:
            ehr_data.append([data.get(key)])
        ehr_data = torch.tensor(ehr_data)
        
        #Prepare labels
        labels= {}
        ga_weeks = data.get('GA')//7
        
        if self.label_smoothing_param is not None and self.train:
            if data.get('relabel'):
                ga_weeks = ga_weeks - self.ga_cutoff
                label_preterm = torch.tensor([1-(self.sigmoid(ga_weeks/self.label_smoothing_param))])
            else:
                label_preterm = torch.tensor([1*self.sigmoid((self.ga_cutoff-ga_weeks)/self.label_smoothing_param)])
        else:
            if data.get('relabel'):
                label_preterm = torch.tensor([1.])
            else:
                label_preterm = torch.tensor([1*(ga_weeks < self.ga_cutoff)])
        
        labels['preterm'] = label_preterm.type(torch.float32)
        labels['GA_reg'] = torch.tensor([float(ga_weeks)])
        
        #Prepare Image       
        img = Image.open(data.get('no_ocr_preprocessed_file_path'))
        img = np.asarray(img)
        img = self.transforms(image=img)['image']

        # try:
        #     img = self.transforms(image=img)['image']
        # except:
        #     print(data.get('no_ocr_preprocessed_file_path'))
        #     img = torch.Tensor(np.zeros((1,224,224)))
        #     labels_temp['preterm'] = torch.Tensor([0])
        #     labels_temp['GA_reg'] = torch.Tensor([0.])
        
        
        #Prepare image metadata
        img_data = []
        for key in self.img_data_vars:
            img_data.append([data.get(key) or 0.0])
        img_data = torch.tensor(img_data)
                
        #Get patient identifier
        cpr = data.get("CPR_CHILD")

        return {'img': img, 'img_data': img_data, 'ehr_data': ehr_data, 'labels': labels, 'cpr': cpr}


def collate_fn(batch):

    label_keys = batch[0]["labels"].keys()

    labels = {key: torch.stack([sample['labels'][key] for sample in batch])
              for key in label_keys}
    
    imgs = torch.stack([sample['img'] for sample in batch])
    img_data = torch.stack([sample['img_data'] for sample in batch])
    ehr_data = torch.stack([sample['ehr_data'] for sample in batch])
    cprs = [sample['cpr'] for sample in batch]


    sample =  {"imgs": imgs,
               "img_data": img_data,
               "ehr_data": ehr_data,
               "labels": labels,
               "cprs": cprs}

    return sample
   

def make_train_val_split(cfg, unique_column='CPR_MOTHER'):
    df = pl.read_parquet(cfg.data.path)
                
    unique_keys = df.select(unique_column).unique()

    rng = np.random.default_rng()
    keys = unique_keys.to_series().to_list()
    rng.shuffle(keys)

    split_idx = int(len(keys) * (1 - cfg.data.val_frac))
    train_keys = keys[:split_idx]
    val_keys = keys[split_idx:]

    train_df = df.filter(pl.col(unique_column).is_in(train_keys))
    val_df = df.filter(pl.col(unique_column).is_in(val_keys))
    
    if cfg.data.oversample:
        df_1 = train_df.filter(pl.col('GA')//7 < cfg.data.ga_cutoff_weeks)
        df_0 = train_df.filter(pl.col('GA')//7 >= cfg.data.ga_cutoff_weeks)
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