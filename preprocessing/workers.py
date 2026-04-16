#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:37:28 2026

@author: jj@di.ku.dk
"""

import sqlite3
import csv
import time
from datetime import datetime

#%% Define worker for preprocess.py
def csv_extracter(path_to_csv, csv_que, done):
    """
    This function loads the CSV info, including the phair_cpr_hash

    Parameters
    ----------
    path_to_csv : str
        path to csv
    csv_que : mp.Queue()
        mp.Queue where we put the extracted csv rows
    done : mp.Value
        shared memory across processes telling the crawlers the csv_extractor is done
    """
    f = open(path_to_csv)
    f_csv = csv.reader(f)
    #Load headers and throw them away
    _ = next(f_csv)

    for row in f_csv:
        csv_que.put(row)
        #Avoid flooding the queue. Not strictly necessary, but preserves memory
        if csv_que.qsize() > 5000:
            time.sleep(1)         
    #Set the shared value true, so the crawlers know no more csv rows are comming
    done.value = True
    f.close()


def db_crawler(csv_idx, db_idx, path_to_db, csv_que, data_que, done):

    con = sqlite3.connect(path_to_db)
    cur = con.cursor()
    while not done.value or csv_que.qsize() > 0:
        row = csv_que.get()
        cpr_mother = row[csv_idx['cpr_mother']]
        cpr_child = row[csv_idx['cpr_child']]
        SHAK = row[csv_idx['Hospital']]
        GA_days = int(row[csv_idx['GA_days']])
        birthdate = datetime.strptime(str(row[csv_idx['Birthdate']]).replace("-",""), "%Y%m%d").date()
        
        query = f"SELECT xxhash FROM cpr_hashes WHERE phair_hash = '{cpr_mother}'"
        cpr_hashes = list(cur.execute(query))
        
        if len(cpr_hashes) == 0:
            data_que.put(['not_found', [cpr_mother, cpr_child, 'Mothers CPR not in DB', SHAK, birthdate]])
        
        elif row[csv_idx['GA_days']] == '.':
            data_que.put(['not_found', [cpr_mother, cpr_child, 'No GA registered', SHAK, birthdate]])
            
        else:    
            data_temp = {}
            
            for key in csv_idx.keys():
                data_temp[key] = row[csv_idx[key]]
    
            imgs_temp = []
                
            for cpr_ in cpr_hashes:
                cpr = cpr_[0]
                try:
                    query = f"SELECT * FROM metadata_cache WHERE file_hash = '{cpr}'"
                    entries = list(cur.execute(query))
                except:
                    data_que.put(['error', [query, 'UTF-8 encoding error in cpr']])
    
                if len(entries) == 0:
                    data_que.put(['error', [str(cpr), 'Empty entry for cprhash']])

                else:    
                    for entry in entries:
                        study_date = entry[db_idx['study_date']]
                        try:
                            study_date = datetime.strptime(str(study_date), "%Y%m%d").date()
                        except:
                            if entry[-1] is not None:
                                data_que.put(['error', [entry[db_idx['file_path']], 'No date or wrong format']])
                                continue
                            else:
                                data_que.put(['error', [entry[db_idx['file_path']], 'Image missing on NGC']])
                                continue
                        
                        diff = ((birthdate - study_date).days)
                        GA_range = GA_days - (birthdate - study_date).days
                        if diff <= 210: #Scan to delivery < 30 weeks
                            if GA_range > 18*7 and GA_range < 39*7: #GA at scan within range
                                img_temp = {}
                                for key in db_idx.keys():
                                    img_temp[key] = entry[db_idx[key]]
                                imgs_temp.append(img_temp)
            imgs = []
            for birth1 in imgs_temp:
                birth_ok = True
                date1 = datetime.strptime(birth1['Birthdate'], "%Y-%m-%d")
                for birth2 in img_temp:
                    if birth1['cpr_child'] == birth2['cpr_child']:
                        continue
                    date2 = datetime.strptime(birth2['Birthdate'], "%Y-%m-%d")
                    if abs((date1-date2).days) <= 40*7:
                        birth_ok = False
                        break
                if birth_ok:
                    imgs.append(birth1)
                        
            if len(imgs) > 0:
                data_temp['imgs'] = imgs
                data_que.put([cpr_child, data_temp])
            else:
                data_que.put(['not_found', [cpr_mother, cpr_child, 'No images associated with birth', SHAK, birthdate]])
