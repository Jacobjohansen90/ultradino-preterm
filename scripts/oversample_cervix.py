#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:15:57 2026

@author: jacob
"""

import polars as pl
import os

seed=12

path_to_train_data = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/train.csv'
path_to_img_data = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/data_dump/img_data.csv'
save_path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/misc/5M_cervix_oversampled.csv'

prefix = '/projects/users/data/UCPH/DeepFetal/ultrasound/PNG_pretrain/'

img_data = pl.read_csv(path_to_img_data, ignore_errors=True)
img_data = img_data[['file_path']].unique()

cv_data = pl.read_csv(path_to_train_data)
cv_data = cv_data[['file_path']].unique()

cv_imgs = cv_data
other_imgs = img_data.filter(pl.col('file_path').is_in(cv_data['file_path']).not_())

cv_sampled = cv_imgs.sample(n=2500000, with_replacement=True, seed=seed)
other_imgs = other_imgs.sample(n=2500000, seed=seed)

cv_sampled = cv_sampled.with_columns((pl.lit(prefix) + pl.col('file_path')).alias('file_path'))
other_imgs = other_imgs.with_columns((pl.lit(prefix) + pl.col('file_path')).alias('file_path'))

final_data = cv_sampled.vstack(other_imgs)

final_data = final_data.filter(pl.col('file_path').map_elements(os.path.isfile))

print(f"Writing {final_data.height} filepaths to csv")

final_data.write_csv(save_path, include_header=False)