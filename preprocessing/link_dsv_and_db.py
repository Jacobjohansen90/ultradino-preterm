#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 12:40:03 2026

@author: jacob
"""

import sqlite3
import csv
from datetime import datetime
import matplotlib.pyplot as plt

variables_from_csv = ['GestationsalderUger',
                      'Alder_Mor']

variables_from_db = ['pixel_spacing_1']

working_dir = '/users/data/UCPH/DeepFetal/projects/preterm/'

con = sqlite3.connect(working_dir + 'Registers/ultrasound_metadata_db.sqlite')
cur = con.cursor()

f = open(working_dir + "Registers/nyfoedte.csv")
f_csv = csv.reader(f)

headers = next(f_csv)

text_cpr = 'CPRnummer_Mor'
text_date = 'FoedselsDato_Barn'

#TODO: Auto infer this index
i_studydate = 5

variables_i = []
days = []

for i in range(len(headers)):
    if headers[i] == text_cpr:
        i_cpr = i
    elif headers[i] == text_date:
        i_date = i
    elif headers[i] in variables_from_csv:
        variables_i.append(i)
        
for row in f_csv:
    cpr_phair = row[i_cpr]
    birthdate = datetime.strftime(str(row[i_date]), "%Y%m%d").date()
    
    query = f"SELECT xxhash FROM cpr_hashes where phair_hash = {cpr_phair}"
    entries = list(cur.execute(query))
    
    for entry in entries:
        study_date = datetime.strftime(str(entry[i_studydate]), "%Y%m%d").date()
        days.append((study_date - birthdate).days)

plt.hist(days, density=True)
plt.savefig(working_dir + 'preprocessing/histogram.png')
        
        
        
        
        
        
    
    
    
    
    
