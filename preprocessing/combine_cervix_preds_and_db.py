#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:27:47 2026

@author: jacob
"""

import csv
import json

preds_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/preprocessing/cervix_preds.csv"
img_link_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/preprocessing/img_cpr_link.json"
data_path =  "/projects/users/data/UCPH/DeepFetal/projects/preterm/preprocessing/data.json"
save_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/preprocessing/cervix_data.json"


f_img_link  = open(img_link_path)
img_link = json.load(f_img_link)

f_data = open(data_path)
data = json.load(f_data)
    
f_preds = open(preds_path)
preds = csv.reader(f_preds)
    
headers = next(preds)
cervix_data = {}

for pred in preds:
    if pred[1] == '14':
        cpr_idx = img_link[pred[0]]
        data = img_link[pred[0]]
        temp = {'cpr_phair_mother': data['cpr_phair_mother'],
                'cpr_phair_child': data['cpr_phair_child'],
                'GA_days': data['GA_days'],
                'GA_weeks': data['GA_days']//7,
                'Age_mother': data['Age_mother']}
        
        for path in data['img_paths']:
            if path[0] == pred[0]:
                temp['img_path'] = path[0]
                temp['ps1'] = path[1]
                temp['ps2'] = path[2]
                temp['scanner'] = path[3]
                
        
        cervix_data[cpr_idx] = temp
        
        
        
f_img_link.close()
f_data.close()
f_preds.close()

with open(save_path, 'w') as f:
    json.dump(cervix_data, f)

