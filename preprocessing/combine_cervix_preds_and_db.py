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
save_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/data/"


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
        data = db_data[img_link[pred[0]]]
        
        if data['GA_days'] == '.':
            wrong_ga.append([img_link[pred[0]], data['GA_days']])
        else:
            temp = {}
            for key in data.keys():
                if key == 'imgs':
                    for img in data[key]:
                        if img['img_path'] == pred[0]:
                            for key in img.keys():
                                temp[key] = img[key]
                            break
                else:
                    temp[key] = data[key]
                               
            cervix_data[pred[0]] = temp
        
f_img_link.close()
f_data.close()
f_preds.close()

with open(save_path + 'cervix_data.json', 'w') as f:
    json.dump(cervix_data, f)

with open(save_path + 'logs/ga_error.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(wrong_ga)

