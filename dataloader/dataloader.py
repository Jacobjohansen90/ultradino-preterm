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
        self.get_segs = 'seg_tasks' in cfg.tasks.keys()
        
        self.aux_vars = []
        for task in cfg.tasks.aux_tasks:
            self.aux_vars.append(task.var)

        self.remove_on_GA_vars = []
        for var, cond in cfg.dataset.items():
            if cond == 'remove_on_GA':
                self.remove_on_GA_vars.append(var)
        
        if self.get_segs:
            self.seg_labels = cfg.tasks.seg_tasks.foreground
        
    
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
        
    def population_count(self, ga_cutoffs):
        population_all = {}
        population_no_prog = {}
        no_prog_df = self.df.filter(~pl.col('progesterone'))

        for cutoff in ga_cutoffs:
            population_all[str(cutoff)] = {'Total Population': self.df["CPR_CHILD"].n_unique(),
                                           'Non-preterm_births': self.df.filter(pl.col("GA")//7 >= cutoff)["CPR_CHILD"].n_unique(),
                                           'Preterm births': self.df.filter((pl.col("GA")//7 < cutoff) 
                                                                     & (pl.all_horizontal(~pl.col(self.remove_on_GA_vars))))["CPR_CHILD"].n_unique()}

            population_no_prog[str(cutoff)] = {'Total Population': no_prog_df["CPR_CHILD"].n_unique(),
                                               'Non-preterm_births': no_prog_df.filter(pl.col("GA")//7 >= cutoff)["CPR_CHILD"].n_unique(),
                                               'Preterm births': no_prog_df.filter((pl.col("GA")//7 < cutoff) 
                                                                            & (pl.all_horizontal(~pl.col(self.remove_on_GA_vars))))["CPR_CHILD"].n_unique()}

        return population_all, population_no_prog
    
    def __len__(self):
        return len(self.df)


    def getitem(self, idx):
                
        #Get data as named dict
        data = self.df.row(idx, named=True)

        #Prepare EHR data
        ehr_data = []
        for key in self.ehr_vars:
            ehr_data.append(float(data.get(key) or 0.0))
        ehr_data = torch.tensor(ehr_data)
        ehr_data = ehr_data.unsqueeze(0)
                
        #Prepare labels
        GA_weeks = data.get('GA')//7
        GA_weeks = torch.tensor([float(GA_weeks)])

        #Prepare auxilary task vars
        aux_vars = {}
        for var in self.aux_vars:
            if var == 'GA_weeks':
                aux_vars[var] = GA_weeks
            else:
                aux_vars[var] = torch.tensor([float(data.get(var) or 0.0)])
                
        #Get segmentations
        if self.get_segs:
            segmentation = np.load(data.get('segmentation_path'))['seg_logits']
            segmentation = np.isin(segmentation, self.seg_labels)
            segmentation = torch.from_numpy(segmentation)
        else:
            segmentation = torch.tensor([0])

        #Prepare remove_on_GA 
        remove_on_GA = torch.tensor([0], dtype=torch.bool)
        for var in self.remove_on_GA_vars:
            if data.get(var):
                remove_on_GA = torch.tensor([1], dtype=torch.bool)
 
        #Prepare Image       
        img = Image.open(data.get('no_ocr_preprocessed_file_path'))
        img = np.asarray(img)
        h = img.shape[0]
        w = img.shape[1]
        img = self.transforms(image=img)['image']

        #Prepare image metadata
        img_data = []
        for key in self.img_data_vars:
            data_temp = data.get(key) or 0.0
            #Correct physical deltas    
            if key == 'physical_delta_x':
                data_temp = (w/self.img_size[0]) * data_temp[0]
            elif key == 'physical_delta_y':
                data_temp = (h/self.img_size[1]) * data_temp[0]
            
            img_data.append(data_temp)
        
        img_data = torch.tensor(img_data)
        img_data = img_data.unsqueeze(0)
                
        #Get patient identifier
        ID = data.get(self.ID_var)
        
        #Get progesterone status
        progesterone = data.get('progesterone')

        return {'img': img, 
                'img_data': img_data, 
                'ehr_data': ehr_data, 
                'GA_weeks': GA_weeks, 
                'ID': ID, 
                'remove_on_GA': remove_on_GA,
                'progesterone': progesterone,
                'aux_vars': aux_vars,
                'segmentation': segmentation}


def collate_fn(batch):
    imgs = torch.stack([sample['img'] for sample in batch])
    img_data = torch.stack([sample['img_data'] for sample in batch])
    ehr_data = torch.stack([sample['ehr_data'] for sample in batch])
    GA_weeks = torch.stack([sample['GA_weeks'] for sample in batch])
    segmentation = torch.stack([sample['segmentation'] for sample in batch])
    IDs = [sample['ID'] for sample in batch]
    remove_on_GA = torch.stack([sample['remove_on_GA'] for sample in batch])
    progesterone = [sample['progesterone'] for sample in batch]
    
    aux_vars = {key: torch.stack([sample['aux_vars'][key] for sample in batch]) for key in batch[0]['aux_vars']}
    

    sample =  {"imgs": imgs,
               "img_data": img_data,
               "ehr_data": ehr_data,
               "GA_weeks": GA_weeks,
               "IDs": IDs,
               "remove_on_GA": remove_on_GA,
               'progesterone': progesterone,
               'aux_vars': aux_vars,
               'segmentation': segmentation}

    return sample
   
    
class DataSplits:
    def __init__(self, cfg, unique_column='CPR_MOTHER', folds=6):
        self.data_path = cfg.data_path
        self.unique_column=unique_column
        self.folds=folds
        self.oversample_ratio = cfg.data.oversample_ratio
        self.highest_GA = max(cfg.tasks.preterm.cutoffs)
        
        train_df = pl.read_parquet(self.data_path + 'train.parquet')
        test_df = pl.read_parquet(self.data_path + 'test.parquet')
        
        for col, cond in cfg.dataset.items():
            if cond == 'remove':
                train_df = train_df.filter(~pl.col(col))
                test_df = test_df.filter(~pl.col(col))
                
        self.train_df = train_df

        # Get the lowest GA for each unique group
        groups = (test_df.group_by(self.unique_column).agg(pl.col("GA").min().alias("GA"))
                  .with_columns((pl.col("GA") // 7).alias("GA_week")))

        # Distribute groups evenly within each GA week
        groups = (groups.with_columns(pl.int_range(pl.len()).shuffle().over("GA_week").alias("fold"))
                  .with_columns((pl.col("fold") % self.folds).alias("fold")).select([self.unique_column, "fold"]))

        # Assign the group's fold to every row
        self.test_df = test_df.join(groups, on=self.unique_column, how="left")
        
        #Check that self.unqiue_column is in exactly one fold
        assert (self.test_df.group_by(self.unique_column).agg(pl.col("fold").n_unique().alias("n_folds"))
                .filter(pl.col("n_folds") != 1).height == 0)
        
        
    def save_distributions(self, save_path):
        fold_counts = (self.test_df.select(["CPR_CHILD", "GA", "fold"])
                       .unique("CPR_CHILD").group_by("fold").agg(pl.len().alias("n"),
                                                                 (pl.col("GA") // 7 < 32).sum().alias("GA < 32"),
                                                                 (pl.col("GA") // 7 < 34).sum().alias("GA < 34"),
                                                                 (pl.col("GA") // 7 < 37).sum().alias("GA < 37")).sort("fold"))
        
        fold_counts.write_csv(save_path + 'GA_distribution.csv')
            
        for fold in range(self.folds):
            test_df_fold = self.test_df.filter(pl.col("fold") == fold)

            test_df_fold.write_parquet(save_path + f"test_df_fold_{fold}.parquet")
        
        
    def get_split(self, fold):
        if not 0 <= fold < self.folds:
            raise Exception(f"Data only contains {self.folds} folds")

        test_df = self.test_df.filter(pl.col('fold') == fold)

        train_df = self.test_df.filter(pl.col('fold') != fold)
        train_df = pl.concat([self.train_df, train_df])

        for col in ["CPR_MOTHER", "CPR_CHILD", "file_path"]:
            overlap = (set(train_df[col].drop_nulls().unique()) & set(test_df[col].drop_nulls().unique()))
            
            if overlap:
                raise ValueError(f"{col}: {len(overlap)} overlaps. \n Examples: {list(overlap)[:10]}")
            
        if self.oversample_ratio != 0:
            positives = train_df.filter(pl.col("GA") // 7 < self.highest_GA)
            negatives = train_df.filter(pl.col("GA") // 7 >= self.highest_GA)
            
            if len(negatives) > len(positives):
                n_target = int(len(negatives) * self.oversample_ratio)
                positives = positives.sample(n=n_target,
                                             with_replacement=True)
                
            elif len(positives) > len(negatives):
                n_target = int(len(positives) * self.oversample_ratio)
                negatives = negatives.sample(n=n_target,
                                             with_replacement=True)
                
            train_df = pl.concat([negatives, positives]) 
        
        
            
        return train_df, test_df
        