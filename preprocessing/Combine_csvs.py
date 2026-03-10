#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:13:34 2026

@author: jacob
"""

import csv

working_path = "/projects/users/data/UCPH/DeepFetal/projects/preterm/Registers/"

headers = ["cpr_child", "cpr_mother", "GA_weeks", "GA_days", "Age_mother", "Birthdate"]

csvs = [["mfr.csv", ["CPR_BARN", "CPR_MODER", None, "GESTATIONSALDER_DAGE", "ALDER_MODER", "FOEDSELSDATO"]],
        ["nyfoedte.csv", ["CPRnummer_Barn", "CPRnummer_Mor", "GestationsalderUger", "Gestationsalder", "Alder_Mor", "FoedselsDato_Barn"]]]

with open(working_path + csv[0]) as file:
    for csv_info in csvs:
        read_file = open(working_path + csv_info[0])
        read_csv = csv.reader(read_file)
        headers = next(read_csv)
        
        for line in read_csv:
            
        