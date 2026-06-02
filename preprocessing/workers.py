#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:37:28 2026

@author: jj@di.ku.dk
"""

import sqlite3
import time
from datetime import datetime
import polars as pl
from collections import Counter

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
        in_que.put(dataframe[i])
        #Avoid flooding the queue. Not strictly necessary, but preserves memory
        if in_que.qsize() > 5000:
            time.sleep(1)         
    #Set the shared value true, so the crawlers know no more csv rows are comming
    done.value = True


def db_crawler(db_idx, path_to_db, in_que, out_que, done):

    con = sqlite3.connect(path_to_db)
    cur = con.cursor()
    while not done.value or in_que.qsize() > 0:
        cpr_mother = in_que.get()
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
                    query = f"SELECT * FROM path_table WHERE file_hash = '{cpr}'"
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
                
def sqlite_extractor(cfg, cpr_mothers):
    chunk_size = 100000
    conn = sqlite3.connect(cfg.paths.SQL_DB)
    cur = conn.cursor()

    #Make a temporary SQL table
    cur.execute("DROP TABLE IF EXISTS tmp_hashes")
    cur.execute("CREATE TEMP TABLE tmp_hashes (phair_hash TEXT PRIMARY KEY)")
    
    #Insert CPR hashes
    insert_sql = "INSERT OR IGNORE INTO tmp_hashes VALUES (?)"
   
    for i in range(0, len(cpr_mothers), chunk_size):
        chunk = cpr_mothers[i:i+chunk_size]
        cur.executemany(insert_sql, ((h,) for h in chunk))
    conn.commit()
    
    dicom_select = ",\n    ".join(f"d.{c}" for c in cfg.metadata_dicom_variables)

    query = f"""
            SELECT
                t.phair_hash,
                pt.file_path,
                pt.file_hash,
                pt.no_ocr_preprocessed_file_path,
                {dicom_select}
            FROM tmp_hashes t
            JOIN cpr_hashes c
                ON c.phair_hash = t.phair_hash
            JOIN path_table pt
                ON pt.file_hash = c.xxhash
            LEFT JOIN dicom_metadata_table d
                ON d.file_hash = pt.file_hash
            """

    cur.execute(query)

    cols = [d[0] for d in cur.description]

    counts = Counter(cols)

    duplicates = [col for col, count in counts.items() if count > 1]

    if duplicates:
        raise ValueError(f"Duplicate columns found: {duplicates}")

    frames = []

    while True:
        rows = cur.fetchmany(chunk_size)
        if not rows:
            break

        clean_rows = [["" if v is None else str(v) for v in row] for row in rows]

        df_chunk = pl.DataFrame(clean_rows, schema=cols, orient="row")

        frames.append(df_chunk)

    df = pl.concat(frames, rechunk=True)     
    
    conn.close()
    
    return df
