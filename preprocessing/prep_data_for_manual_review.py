#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:58:11 2026

@author: jacob
"""

import pandas as pd
import numpy as np
import shutil
import json

preds_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/preprocessing/cervix_preds.csv"
save_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/preprocessing/emilie_check/"

df = pd.read_csv(preds_path)

df = df[df['pred'] == 14]

df = df.reindex(np.random.permutation(df.index))

df = df.head(1000)

data = df.to_numpy()

linker = {}
i = 1
for entry in data:
    shutil.copyfile(entry[0], save_path + str(i).zfill(4) + '.png')
    linker[str(i).zfill(4) + '.png'] = entry[0]

with open(save_path + 'linker.json', 'w') as file:
    json.dump(linker, file)

