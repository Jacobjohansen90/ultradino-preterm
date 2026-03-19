#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:29:55 2026

@author: jacob
"""
#%%Imports
import csv
import os
import json 
import logging
import multiprocessing as mp
from pathlib import Path
import pandas as pd
import shutil

from utils import csv_extracter, db_crawler
from inference_classification import infer
#%%Variables

#Path to Data folder and holdout test set
path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/"
holdout_path = "/projects/users/data/UCPH/DeepFetal/projects/ultrasound_preprocessing/splits/PRETERM_RCT_PT_V2/holdout_test_5pct.csv"

#Variables
force_overwrite = False #Force complete preprocess

num_workers = 60 #Number of MP workers

#CSV Variables we want from registeres in each file
headers = ["cpr_child", "cpr_mother", "GA_days", "Age_mother", "Birthdate"]

csvs = [["mfr.csv", ["CPR_BARN", "CPR_MODER", "GESTATIONSALDER_DAGE", "ALDER_MODER", "FOEDSELSDATO"]],
        ["nyfoedte.csv", ["CPRnummer_Barn", "CPRnummer_Mor", "Gestationsalder", "Alder_Mor", "FoedselsDato_Barn"]]]


#CSV indexes we want in the final output
variables_from_csv = ['GA_days',
                      'Age_mother',
                      'cpr_child',
                      'cpr_mother',
                      'Birthdate']

#Sqlite database indexes we want in the final output
variables_from_db = ['file_path',
                     'no_ocr_preprocessed_file_path',
                     'manufacturer',
                     'manufacturer_model',
                     'study_date',
                     'physical_delta_x',
                     'physical_delta_y']




#%%Combine CSVs

n_csv = 0

if 'data.csv' not in os.listdir(path + 'registers') or force_overwrite:
    with open(path + 'data.csv', 'w') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(headers)
    
        for csv_info in csvs:
            idxs = []
            csv_file = open(working_path + csv_info[0])
            csv_headers = csv_info[1]
            csv_ = csv.reader(csv_file)
            temp_headers = next(csv_)
            
            for head in csv_headers:
                i = 0
                for temp_head in temp_headers:
                    if temp_head == head:
                        idxs.append(i)
                        break
                    else:
                        i += 1
                    
            for row in csv_:
                info = []
                for idx in idxs:
                    n_csv += 1
                    info.append(row[idx])
                wr.writerow(info)
        
        csv_file.close()

#%%Crawl database

#Setup ques, loggers and start processes
csv_que = mp.Queue()
data_que = mp.Queue()
done = mp.Value('b', False)
csv_size = mp.Value('i', 0)
path_to_db = path + 'registers/ultrasound_metadata_db.sqlite'
csv_idx = {}

#Crawl CSV for variable indexes
if n_csv == 0:
    f = open(path + 'registers/data.csv')
    f_csv = csv.reader(f)
    csv_headers = next(f_csv)
    csv_size.value = sum(1 for line in f_csv)
    f.close()

else:
    csv_size.value = n_csv

for i in range(len(csv_headers)):
    for variable in variables_from_csv:
        if headers[i] == variable:
            csv_idx[variable] = i

if len(variables_from_csv) != len(csv_idx):
    found = list(csv_idx.keys())
    diff = list(set(variables_from_csv) - set(found))
    raise Exception(f"Did not find variables {diff} in CSV")

#Crawl DB for variables indexes
with sqlite3.connect(path_to_db) as con:
    cur = con.cursor()
    cursor.execute("SELECT * FROM metadata_cache LIMIT 0")
    db_headers = [desc[0] for desc in cursor.description]

for i in range(len(db_headers)):
    for variable in variables_from_db:
        if headers[i] == variable:
            db_idx[variable] = i


logging.basicConfig(filename=path + 'preprocess.log', filemode='w')
logger = logging.getLogger('Preprocess')
logger.setLevel(logging.INFO)

num_workers = min(num_workers, mp.cpu_count()-4)

logger.info(f"Starting {num_workers} workers - " + str(datetime.now().strftime('%H:%M:%S')))

processes = []
p = mp.Process(target=csv_extracter, args=(path + 'registers/data.csv', csv_que, done))
p.start()
processes.append(p)

for i in range(num_workers):
    p = mp.Process(target=db_crawler, args=(csv_idx, db_idx, path_to_db, csv_que, data_que, done))
    p.start()
    processes.append(p)

not_found = []
errors = []
final_data = {}
img_cpr_link = {}
invalid_counter = 0
n = 1
images = []

while n < csv_size.value:

    data = data_que.get()

    if data[0] == 'error':
        errors.append(data[1])
    elif data[0] == 'not_found':
        n += 1
        not_found.append(data[1])
        if n % 10000 == 0:
            logger.info(f"Completed {n} files - " + str(datetime.now().strftime('%H:%M:%S')))
    else:
        if data[0] == 'INVALID':
            final_data['child_' + str(invalid_counter)] = data[1]
            invalid_counter += 1
        else:
            final_data[data[0]] = data[1]
        n += 1
        if n % 10000 == 0:
            logger.info(f"Completed {n} files - " + str(datetime.now().strftime('%H:%M:%S')))            

Path(path + 'logs/').mkdir(exist_ok=True)
Path(path + 'image_data/misc/').mkdir(parents=True, exist_ok=True)

with open(path + 'logs/missing.csv', 'w', newline='') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    wr.writerow(["cpr_phair_mother", "cpr_phair_child", "error"])
    for row in not_found:
        wr.writerow(row)

with open(path + 'logs/errors.csv', 'w', newline='') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    wr.writerow(["info", "error"])
    for row in errors:
        wr.writerow(row)

with open(path + 'image_data/all_data.json', 'w') as file:
    json.dump(final_data, file)
    
with open(path + 'image_data/misc/image_list.csv', 'w') as file:
    wr = csv.writer(file)
    wr.writerow(["filename"])
    for key in final_data.keys():
        for img_path in final_data[key]['imgs']:
            path = img_path['img_path']
            images.append(path)
            wr.writerow([path])
            img_cpr_link[path] = key
        
with open(save_path + 'image_data/img_cpr_link.json', 'w') as file:
    json.dump(img_cpr_link, file)
    
del not_found
    
#%%Do cervix prediction

if os.path.exists(path + 'image_data/misc/cervix_preds.csv') and not force_overwrite:    
    f = open(path + 'image_data/misc/cervix_preds.csv')
    reader = csv.reader(f)
    headers = next(reader)
    new_images = [path for path in images not in set(list(reader))]
    f.close()
    
    f = open(path + 'image_data/misc/cervix_check.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(["filename"])
    for path in new_images:
        writer.writerow([path])
        
    f.close()
    

else:
    shutil.copyfile(path + 'image_data/misc/image_list.csv', path + 'image_data/misc/cervix_check.csv')
        
del images
infer()

if os.path.exists(path + 'image_data/misc/cervix_preds.csv') and not force_overwrite:    
    df1 = pd.read_csv(path + 'image_data/misc/cervix_preds_temp.csv')
    df1.to_csv(path + 'image_data/misc/cervix_preds.csv', mode='a', header=False, index=False)
    os.remove(path + 'image_data/misc/cervix_preds_temp.csv')
else:
    os.rename(path + 'image_data/misc/cervix_preds_temp.csv', path + 'image_data/misc/cervix_preds.csv')

#%%Remove test set from data  
f_preds = open(path + 'cervix_preds.csv')
preds = csv.reader(f_preds)

f_holdout = open(holdout_path)  
holdout_csv = csv.reader(f_holdout)  

headers = next(preds)

cervix_data_all = {}
cervix_data = {}
holdout_data = {}
no_ga = []

holdout_set = set(list(holdout_csv))

for pred in preds:
    if pred[1] == '14':
        data = db_data[img_link[pred[0]]]
        
        if data['GA_days'] == '.':
            no_ga.append([img_link[pred[0]]])
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
            if pred[0] in holdout:
                holdout_data[pred[0]] = temp
                cervix_data_all[pred[0]] = temp
            else:
                cervix_data[pred[0]] = temp
                cervix_data_all[pred[0]] = temp

f_preds.close()
f_holdout.close()

with open(path + 'image_data/misc/cervix_data_all.json', 'w') as f:
    json.dump(cervix_data_all, f)

with open(path + 'image_data/holdout_data.json', 'w') as f:
    json.dump(holdout_data, f)

with open(path + '/image_data/cervix_data.json', 'w') as f:
    json.dump(cervix_data, f)


with open(path + 'logs/ga_missing.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(wrong_ga)

