#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:37:28 2026

@author: jj@di.ku.dk
"""

import sqlite3
import time
from datetime import datetime

#%% Define worker for preprocess.py
def csv_extracter(dataframe, in_que, done):
    """
    This function loads the CSV info, including the phair_cpr_hash

    Parameters
    ----------
    dataframe : Polars dataframe
        Polars dataframe with mothers CPRs
    csv_que : mp.Queue()
        mp.Queue where we put the extracted csv rows
    done : mp.Value
        shared memory across processes telling the crawlers the csv_extractor is done
    """
    dataframe = dataframe['CPR_MOTHER'].unique()
    for i in range(len(dataframe)):
        in_que.put(dataframe[0])
        #Avoid flooding the queue. Not strictly necessary, but preserves memory
        if in_que.qsize() > 5000:
            time.sleep(1)         
    #Set the shared value true, so the crawlers know no more csv rows are comming
    done.value = True


def db_crawler(db_idx, path_to_db, in_que, out_que, done):

    con = sqlite3.connect(path_to_db)
    cur = con.cursor()
    while not done.value or in_que.qsize() > 0:
        cpr_mother = in_que.get()[0]
        temp_dict = {'CPR_MOTHER': cpr_mother}
        query = f"SELECT xxhash FROM cpr_hashes WHERE phair_hash = '{cpr_mother}'"
        cpr_hashes = list(cur.execute(query))
        
        if len(cpr_hashes) == 0:
            out_que.put(['CPR_error', 'Mothers CPR not in DB', cpr_mother])
        
        else:                
            imgs = []
            
            for cpr_ in cpr_hashes:
                cpr = cpr_[0]
                try:
                    query = f"SELECT * FROM metadata_cache WHERE file_hash = '{cpr}'"
                    entries = list(cur.execute(query))
                except:
                    out_que.put(['DB_error', 'UTF-8 encoding error in cpr', query])
    
                if len(entries) == 0:
                    out_que.put(['DB_error', 'Empty entry for cprhash', query])

                else:    
                    for entry in entries:
                        study_date = entry[db_idx['study_date']]
                        try:
                            study_date = datetime.strptime(str(study_date), "%Y%m%d").date()
                        except:
                            if entry[-1] is not None:
                                out_que.put(['img_error', 'No date or wrong format', entry[db_idx['file_path']]])
                                continue
                            else:
                                out_que.put(['img_error', 'Image missing on NGC', entry[db_idx['file_path']]])
                                continue
                        img_temp = {}
                        for key in db_idx.keys():
                            img_temp[key] = entry[db_idx[key]]
                        imgs.append(img_temp) 
                                                
            if len(imgs) > 0:
                temp_dict['imgs'] = imgs
                out_que.put([cpr_mother, temp_dict])
            else:
                out_que.put(['CPR_error', 'No images associated with CPR', cpr_mother])
