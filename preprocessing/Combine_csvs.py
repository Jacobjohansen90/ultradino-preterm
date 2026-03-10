#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:13:34 2026

@author: jacob
"""

import csv

working_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/Registers/"

headers = ["cpr_child", "cpr_mother", "GA_days", "Age_mother", "Birthdate"]

csvs = [["mfr.csv", ["CPR_BARN", "CPR_MODER", "GESTATIONSALDER_DAGE", "ALDER_MODER", "FOEDSELSDATO"]],
        ["nyfoedte.csv", ["CPRnummer_Barn", "CPRnummer_Mor", "Gestationsalder", "Alder_Mor", "FoedselsDato_Barn"]]]

with open(working_path + 'data.csv') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    wr.writerow(headers)

    for csv_info in csvs:
        idxs = []
        csv_file = open(working_path + csv_info[0])
        csv_headers = csv_info[1]
        csv = csv.reader(csv_file)
        temp_headers = next(csv)
        
        for head in csv_headers:
            i = 0
            for temp_head in temp_headers:
                if temp_head == head:
                    idxs.append(i)
                    break
                else:
                    i += 1
                
        for row in csv:
            info = []
            for idx in idxs:
                info.append(row[i])
            wr.writerow(info)
    
    csv_file.close()
    