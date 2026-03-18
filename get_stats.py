#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:22:13 2026

@author: jacob
"""

import json
import csv

path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/'
stats = {}
#%%Count births
f = open(path + 'registers/data.csv')
d = csv.reader(f)

headers = next(d)

births = sum(1 for line in d)

stats['total_births'] = births

f.close()

#%%Count DB
f = open(path + 'data/all_data.json')
d = json.load(f)

stats['n_in_database'] = len(d)

f.close()

f = open(path + 'data/logs/missing.csv')
d = csv.reader(f)

headers = next(d)

missing = sum(1 for line in d)

stats['n_missing_in_database'] = missing

f.close()
#%%Count images
f = open(path + 'data/image_list.csv')
d = csv.reader(f)

headers = next(d)

imgs = sum(1 for line in d)

stats['images'] = imgs

f.close()
#%%Count errors
f = open(path + 'data/logs/errors.csv')
d = csv.reader(f)

headers = next(d)

counter = {}

for row in d:
    if row[1] not in stats.keys():
        stats[row[1]] = 1
    else:
        stats[row[1]] += 1


#%%Count cervix

f = open(path + 'data/cervix_data.json')
d = json.load(f)

stats['has_cervix'] = len(d)

f.close()

f = open(path + 'data/logs/ga_error.csv')
d = csv.reader(f)

missing = sum(1 for line in d)

stats['missing_GA'] = missing

f.close()

#%%Write to file

f = open(path + 'stats.txt', 'w')
for key, stat in stats.items():
    f.write(f"{key} : {stat}")

