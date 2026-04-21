#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:04:44 2026

@author: jacob
"""

import polars as pl
import shutil
from pathlib import Path
import json

n = 1000
seed=12
prefix = '/projects/users/data/UCPH/DeepFetal/ultrasound/PNG_pretrain/'
save_path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/misc/emilie_check2/'

train_data = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/train.csv'

img_data = pl.read_csv(train_data)

img_data = img_data[['file_path']].unique()
sample = img_data.sample(n=n, with_replacement=False, seed=seed)

Path(save_path).mkdir(parents=True, exist_ok=True)

linker = {}

for i, path in enumerate(sample['file_path']):
    path = prefix + path
    shutil.copy(path, save_path + str(i+1).zfill(4))
    linker[str(i+1).zfill(4)] = path
    
with open(save_path + 'linker.json', 'w') as file:
    json.dump(linker, file)
    

