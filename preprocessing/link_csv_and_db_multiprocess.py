#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Thu Mar 12 14:42:12 2026

@author: jacob
"""

#%% Imports
import sqlite3
import csv
from datetime import datetime
import json 
import logging
import multiprocessing as mp
import time

#%% Variables

#Number of threads for MP
num_workers = 60

#Base directory
path_to_csv = '/projects/users/data/UCPH/DeepFetal/projects/preterm/registers/data.csv'
path_to_db = '/projects/users/data/UCPH/DeepFetal/projects/preterm/registers/ultrasound_metadata_db.sqlite'
save_path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/data/'

#CSV indexes we want in the final output
variables_from_csv = ['GA_days',
                      'Age_mother',
                      'cpr_child',
                      'cpr_mother',
                      'Birthdate']

#Sqlite database indexes we want in the final output
db_idx = {'img_path': -1,
          'sonai_path': 0,
          'manufactor': 3,
          'scanner_type': 4,
          'studydate': 5,
          'pdx': 8,
          'pdy': 9}


#%% Define worker functions

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
        #Avoid flooding the queue. Not strictly necessary, but preserve memory
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

        birthdate = datetime.strptime(str(row[csv_idx['Birthdate']]).replace("-",""), "%Y%m%d").date()
        
        query = f"SELECT xxhash FROM cpr_hashes WHERE phair_hash = '{cpr_mother}'"
        cpr_hashes = list(cur.execute(query))
        
        if len(cpr_hashes) == 0:
            data_que.put(['not_found', [cpr_mother, cpr_child, 'no_cpr_link_mother']])
        
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
                    data_que.put(['error', [query, 'Query - UTF-8 encoding error']])
    
                if len(entries) == 0:
                    data_que.put(['error', [str(cpr), 'CPR - no_data_for_xxhash']])

                else:    
                    for entry in entries:
                        study_date = entry[db_idx['studydate']]
                        try:
                            study_date = datetime.strptime(str(study_date), "%Y%m%d").date()
                        except:
                            if entry[-1] is not None:
                                data_que.put(['error', [entry[db_idx['img_path']], 'Img_path - date_not_found_or_wrong_format']])
                                continue
                            else:
                                data_que.put(['error', [entry[db_idx['sonai_path']], 'Img_path - image_missing_on_NGC']])
                                continue
                        
                        if abs((study_date - birthdate).days) < 280:
                            if entry[db_idx['img_path']] is None:
                                data_que.put(['error', [entry[db_idx['sonai_path']], 'Img_path - image_missing_on_NGC']])
                            else:
                                img_temp = {}
                                for key in db_idx.keys():
                                    img_temp[key] = entry[db_idx[key]]
                                imgs.append(img_temp)
    
            if len(imgs) > 0:
                data_temp['img_paths'] = imgs
                data_que.put([cpr_child, data_temp])
            else:
                data_que.put(['not_found', [cpr_mother, cpr_child, 'no_imgs_for_child']])


#%% Setup ques, loggers and start processes
csv_que = mp.Queue()
data_que = mp.Queue()
done = mp.Value('b', False)
csv_size = mp.Value('i', 0)

csv_idx = {}


f = open(path_to_csv)
f_csv = csv.reader(f)
headers = next(f_csv)
csv_size.value = sum(1 for line in f_csv)
f.close()

for i in range(len(headers)):
    for variable in variables_from_csv:
        if headers[i] == variable:
            csv_idx[variable] = i

if len(variables_from_csv) != len(csv_idx):
    found = list(csv_idx.keys())
    diff = list(set(variables_from_csv) - set(found))
    raise Exception(f"Did not find variables {diff} in CSV")


logging.basicConfig(filename=save_path + 'logs/database_linker.log', filemode='w')
logger = logging.getLogger('link_csv_and_db')
logger.setLevel(logging.INFO)

num_workers = min(num_workers, mp.cpu_count()-4)

logger.info(f"Starting {num_workers} workers - " + str(datetime.now().strftime('%H:%M:%S')))

processes = []
p = mp.Process(target=csv_extracter, args=(path_to_csv, csv_que, done))
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

while n < csv_size.value:

    data = data_que.get()

    if data[0] == 'error':
        errors.append(data[1])
    elif data[0] == 'not_found':
        n += 1
        not_found.append(data[1])
        if n % 1000 == 0:
            logger.info(f"Completed {n} files - " + str(datetime.now().strftime('%H:%M:%S')))
    else:
        final_data[data[0]] = data[1]
        n += 1
        if n % 1000 == 0:
            logger.info(f"Completed {n} files - " + str(datetime.now().strftime('%H:%M:%S')))            

with open(save_path + 'logs/missing.csv', 'w', newline='') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    wr.writerow(["cpr_phair_mother", "cpr_phair_child", "error"])
    for row in not_found:
        wr.writerow(row)

with open(save_path + 'logs/errors.csv', 'w', newline='') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    wr.writerow(["info", "error"])
    for row in errors:
        wr.writerow(row)

with open(save_path + 'database_crawl.json', 'w') as file:
    json.dump(final_data, file)
    
with open(save_path + 'image_list.csv', 'w') as file:
    wr = csv.writer(file)
    wr.writerow(["filename"])
    for key in final_data.keys():
        for img_path in final_data[key]['img_paths']:
            path = img_path[0]
            wr.writerow([path])
            img_cpr_link[path] = key
        
with open(save_path + 'img_cpr_link.json', 'w') as file:
    json.dump(img_cpr_link, file)
        
















        


