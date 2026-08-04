#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:15:57 2026

@author: jacob
"""

import polars as pl

holdout_csv = '/projects/users/data/UCPH/DeepFetal/projects/common/splits/split_V4/test_split_0.15_2026-06-08.csv'
img_data = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/AnyPreg_June_v3/data_dump/img_data.csv'
save_path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/misc/cervix.csv'

df_hold = pl.read_csv(holdout_csv)
df = pl.read_csv(img_data)

df = df.filter(pl.col("pred") == 14)

df = df.join(df_hold, left_on='CPR_MOTHER', right_on='CPR_MOR', how='anti')

df = df['no_ocr_preprocessed_file_path']

df.write_csv(save_path, include_header=False)