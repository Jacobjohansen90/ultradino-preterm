#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 12:40:03 2026

@author: jacob
"""

import sqlite3
import csv
from datetime import datetime
import numpy as np
import json 
import logging

logging.basicConfig(filename="/projects/users/data/UCPH/DeepFetal/projects/preterm/log.log", filemode='w')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

variables_from_csv = ['GestationsalderUger',
                      'Alder_Mor']

variables_from_db = [6,7,-1]

working_dir = '/projects/users/data/UCPH/DeepFetal/projects/preterm/'

con = sqlite3.connect(working_dir + 'Registers/ultrasound_metadata_db.sqlite')
cur = con.cursor()

f = open(working_dir + "Registers/nyfoedte.csv")
f_csv = csv.reader(f)

headers = next(f_csv)

text_cpr = 'CPRnummer_Mor'
text_date = 'FoedselsDato_Barn'

#TODO: Auto infer this index
i_studydate = 5

csv_variables_i = []
not_found = []
imgs = {}
for i in range(len(headers)):
    if headers[i] == text_cpr:
        i_cpr = i
    elif headers[i] == text_date:
        i_date = i
    elif headers[i] in variables_from_csv:
        csv_variables_i.append(i)

if len(variables_from_csv) != len(csv_variables_i):
    found = ([headers[i] for i in csv_variables_i])
    diff = list(set(found) - set(variables_from_csv))
    raise Exception(f"Did not variables {diff} in CSV")
        
n = 0
for row in f_csv:
    n += 1
    if n % 1000 == 0:
        logger.info(f"Completed {i} files")
    
    cpr_phair = row[i_cpr]
    birthdate = datetime.strptime(str(row[i_date]).replace("-",""), "%Y%m%d").date()
    
    query = f"SELECT xxhash FROM cpr_hashes WHERE phair_hash = '{cpr_phair}'"
    cpr_hash = list(cur.execute(query))
    
    if len(cpr_hash) == 0:
        not_found.append([cpr_phair, 'no_cpr_link'])
    else:    
        temp = {'cpr_phair': cpr_phair}
        for i in range(len(csv_variables_i)):
            temp[variables_from_csv[i]] = row[csv_variables_i[i]]
        cpr_hash = cpr_hash[0][0]
        query = f"SELECT * FROM metadata_cache WHERE file_hash = '{cpr_hash}'"
        entries = list(cur.execute(query))
        img_paths = []
        for entry in entries:
            study_date = entry[i_studydate]
            if study_date is None:
                not_found.append([cpr_hash, 'no_date'])
            else:
                study_date = datetime.strptime(str(study_date), "%Y%m%d").date()
                if np.abs((study_date - birthdate).days) < 280:
                    ps1 = entry[6]
                    ps2 = entry[7]
                    img_path = entry[-1]
                    img_paths.append([img_path, ps1, ps2])
            imgs[cpr_hash] = temp
            imgs['imgs'] = img_paths
        
with open(working_dir + 'preprocessing/missing.csv', 'w', newline='') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    for row in not_found:
        wr.writerow(row)


with open(working_dir + 'preprocessing/data.json', 'w') as file:
    json.dump(imgs, file)
        
        
        
        
    
    
    
    
    
