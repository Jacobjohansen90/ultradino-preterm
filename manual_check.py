#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:32:07 2026

@author: jacob
"""
import pandas as pd
import shutil
from pathlib import Path
import csv

path = '/projects/users/data/UCPH/DeepFetal/preterm/Data/image_data/misc/cervix_preds.csv'
save_path = '/projects/users/data/UCPH/DeepFetal/preterm/Data/check/rename_me/'
prefix = '/projects/users/data/UCPH/DeepFetal/ultrasound/PNG_pretrain/'

Path(save_path).mkdir(exist_ok=True)

n_imgs = 1000

df = pd.read_csv(path)
df = df[df['pred'] == 14]
df = df.sample(n=n_imgs)

linker = {}

with open(save_path + 'img_paths.csv') as file:
    wr = csv.writer(file)
    wr.writerow(['index', 'filepath'])
    for i in range(n_imgs):
        file_path = prefix + df.iloc[i].filepath
        shutil.copy(file_path, save_path + str(i+1).zfill(4) + '.png')
        linker[str(i+1).zfill(4)] = file_path
        wr.writerow([str(i+1).zfill(4), file_path])
        
        