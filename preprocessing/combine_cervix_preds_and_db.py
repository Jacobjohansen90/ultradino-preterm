#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:27:47 2026

@author: jacob
"""

import csv
import json

path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/data/"

preds_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/data/cervix_preds.csv"
img_link_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/data/img_cpr_link.json"
data_path =  "/projects/users/data/UCPH/DeepFetal/projects/preterm/data/all_data.json"
save_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/preprocessing/"


f_img_link  = open(path + 'img_cpr_link.json')
img_link = json.load(f_img_link)

f_data = open(path + 'all_data.json')
db_data = json.load(f_data)
    
f_preds = open(path + 'cervix_preds.csv')
preds = csv.reader(f_preds)
    
headers = next(preds)

cervix_data = {}
wrong_ga = []

for pred in preds:
    if pred[1] == '14':
        cpr_idx = img_link[pred[0]]
        data = db_data[cpr_idx]
        
        try:
            temp = {'cpr_mother': data['cpr_mother'],
                    'cpr_child': data['cpr_child'],
                    'GA_days': int(data['GA_days']),
                    'GA_weeks': int(data['GA_days'])//7,
                    'age_mother': data['Age_mother'],
                    'birthday': data['birthday']}
            
            for img in data['imgs']:
                if img['img_path'] == pred[0]:
                    for key in img.keys():
                        temp[key] = img[key]
                    break
                               
            cervix_data[cpr_idx] = temp

        except:
            wrong_ga.append([cpr_idx, data['GA_days']])
        
f_img_link.close()
f_data.close()
f_preds.close()

with open(save_path + 'cervix_data.json', 'w') as f:
    json.dump(cervix_data, f)

with open(save_path + 'ga_error.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(wrong_ga)

