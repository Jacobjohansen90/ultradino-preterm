#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:22:13 2026

@author: jacob
"""

import json
import csv

path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/'

stats = open(path + 'stats.txt')

#%%Count births
f = open(path + 'Data/registers/combined.csv')
d = csv.reader(f)

headers = next(d)
   
births = sum(1 for line in d)
stats.write('--Total Births--\n')
stats.write('Total births in SDS: ' + str(births) + '\n')

f.close()

#%%Count DB
f = open(path + 'Data/image_data/img_data.json')
d = json.load(f)

births_in_sql = len(d)

stats.write('Total births in SQL db: '+ str(births_in_sql) + '\n')

f.close()

f = open(path + 'data/logs/missing.csv')
d = csv.reader(f)

headers = next(d)

missing = 0

counter = {}

for row in d:
    missing += 1
    if row[2] not in counter.keys():
        stats[row[2]] = 1
    else:
        stats[row[2]] += 1

stats.write('Total births missing in SQL db: ' + str(missing) + '\n')
for key in counter:
    stats.write('\t- ' + str(key) + ' :' + str(counter[key]) + '\n')
    
uacc = births - births_in_sql - missing
stats.write('Unaccounted for: ' + str(uacc) + '\n')

f.close()

#%%Count images
stats.write('\n')
stats.write('--Images--\n')

f = open(path + 'Data/image_data/misc/image_list.csv')
d = csv.reader(f)

headers = next(d)

imgs = sum(1 for line in d)

stats.write('Total images: ' + str(imgs) + '\n')

f.close()
#%%Count errors
f = open(path + 'Data/logs/errors.csv')
d = csv.reader(f)

headers = next(d)

counter = {}

for row in d:
    if row[1] not in stats.keys():
        counter[row[1]] = 1
    else:
        counter[row[1]] += 1
        
for key in counter:
    stats.write('\t- ' + str(key) + ' :' + str(counter[key]) + '\n')       

#%%Count cervix TBD

# f = open(path + 'data/cervix_data.json')
# d = json.load(f)

# stats['is_cervix'] = len(d)

# f.close()

# f = open(path + 'data/logs/ga_error.csv')
# d = csv.reader(f)

# missing = sum(1 for line in d)

# stats['missing_GA'] = missing

# f.close()

#%%Regional + hospital breakdown
stats.write('\n')
stats.write('--Regional + Hospital breakdown--\n')
stats.write('\n')

f = open(path + 'Data/registers/nyfoedte.csv')
d = csv.reader(f)

headers = next(d)

n_reg = {}
n_hos = {}

for i, header in enumerate(headers):
    if header == 'AnsvarligRegion_Geo_Tekst':
        reg_txt = i
    elif header == 'AnsvarligInstitution_Kode':
        hos_kode = i
    elif header == 'AnsvarligInstitution_Tekst':
        hos_txt = i

translator = {}

for line in d:
    if line[hos_kode] not in translator.keys():
        translator[hos_kode] = [line[hos_txt], line[reg_txt]]

f.close()

f = open(path + 'Data/image_data/img_data.json')
cprs = json.load(f)

cpr_child = set(cprs.keys())

g = open(path + 'Data/registers/combined.csv')
d = csv.reader(g)

_ = next(d)

for line in d:
    if line[0] in cpr_child:
        reg, hos = translator[line[4]]
        if reg not in n_reg.keys():
            n_reg[reg] = 1
        else:
            n_reg[reg] += 1
        if hos not in n_hos.keys():
            n_hos[hos] = 1
        else:
            n_hos[hos] += 1

stats.write('Birth with images\n')
for key in n_reg:
    stats.write('\t- ' + str(key) + ' :' + str(counter[key]) + '\n')
stats.write('\n')
for key in n_hos:
    stats.write('\t- ' + str(key) + ' :' + str(counter[key]) + '\n')
 
