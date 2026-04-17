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
        
        query = f"SELECT xxhash FROM cpr_hashes WHERE phair_hash = '{cpr_mother}'"
        cpr_hashes = list(cur.execute(query))
        
        if len(cpr_hashes) == 0:
            data_que.put(['not_found', [cpr_child, cpr_mother, 'Mothers CPR not in DB']])
        
        else:
            data_temp = {}
            
            for key in csv_idx.keys():
                data_temp[key] = row[csv_idx[key]]
    
            imgs = []
                
            for cpr_ in cpr_hashes:
                cpr = cpr_[0]
                try:
                    query = f"SELECT * FROM metadata_cache WHERE file_hash = '{cpr}'"
                    entries = list(cur.execute(query))
                except:
                    data_que.put(['error', [query, 'UTF-8 encoding error in cpr']])
    
                if len(entries) == 0:
                    data_que.put(['error', [query, 'Empty entry for cprhash']])

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
                                                
            if len(imgs) > 0:
                data_temp['imgs'] = imgs
                data_que.put([cpr_child, data_temp])
            else:
                data_que.put(['not_found', [cpr_child, cpr_mother, 'No images associated with birth']])
