#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:42:12 2026

@author: jacob
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 12:40:03 2026

@author: jacob
"""

import sqlite3
import csv
from datetime import datetime
import json 
import logging
import multiprocessing as mp
import time

num_workers = 60

csv_que = mp.Queue()
error_que = mp.Queue()
not_found_que = mp.Queue()
data_que = mp.Queue()
worker_done_que = mp.Queue()

logging.basicConfig(filename="/projects/users/data/UCPH/DeepFetal/projects/preterm/preprocess.log", filemode='w')

logger = logging.getLogger('link_csv_and_db')
logger.setLevel(logging.INFO)

variables_from_csv = ['GA_days',
                      'Age_mother']

working_dir = '/projects/users/data/UCPH/DeepFetal/projects/preterm/'

f = open(working_dir + "Registers/data.csv")
f_csv = csv.reader(f)

headers = next(f_csv)

child_cpr = 'cpr_child'
mother_cpr = 'cpr_mother'
text_date = 'Birthdate'

csv_variables_i = []

for i in range(len(headers)):
    if headers[i] == mother_cpr:
        mother_cpr_i = i
    elif headers[i] == child_cpr:
        child_cpr_i = i
    elif headers[i] == text_date:
        i_date = i
    elif headers[i] in variables_from_csv:
        csv_variables_i.append(i)

if len(variables_from_csv) != len(csv_variables_i):
    found = ([headers[i] for i in csv_variables_i])
    diff = list(set(found) - set(variables_from_csv))
    raise Exception(f"Did not find variables {diff} in CSV")

f.close()

#TODO: Auto infer this index
i_studydate = 5
variables_from_db = [3,4,6,7,-1]

def worker(mother_cpr_i, 
           child_cpr_i, 
           i_date, variables_from_csv, 
           i_studydate, 
           variables_from_db, 
           working_dir, 
           csv_que, 
           error_que, 
           not_found_que, 
           data_que, 
           worker_done_que):
    
    con = sqlite3.connect(working_dir + 'Registers/ultrasound_metadata_db.sqlite')
    cur = con.cursor()
    while True:
        row = csv_que.get()
        cpr_phair_mother = row[mother_cpr_i]
        cpr_phair_child = row[child_cpr_i]
        birthdate = datetime.strptime(str(row[i_date]).replace("-",""), "%Y%m%d").date()
        query = f"SELECT xxhash FROM cpr_hashes WHERE phair_hash = '{cpr_phair_mother}'"
        cpr_hashes = list(cur.execute(query))
        if len(cpr_hashes) == 0:
            not_found_que.put([cpr_phair_mother, cpr_phair_child, 'no_cpr_link_mother'])
        else:    
            temp = {'cpr_phair_mother': cpr_phair_mother}
            temp['cpr_phair_child'] = cpr_phair_child
    
            for i in range(len(csv_variables_i)):
                temp[variables_from_csv[i]] = row[csv_variables_i[i]]
    
            img_paths = []
    
            for cpr_ in cpr_hashes:
                cpr = cpr_[0]
                try:
                    query = f"SELECT * FROM metadata_cache WHERE file_hash = '{cpr}'"
                    entries = list(cur.execute(query))
                except:
                    error_que.put([query, 'Query - UTF-8 encoding error'])
    
                if len(entries) == 0:
                    error_que.put([str(cpr), 'CPR - no_data_for_xxhash'])
                else:    
                    for entry in entries:
                        study_date = entry[i_studydate]
                        try:
                            study_date = datetime.strptime(str(study_date), "%Y%m%d").date()
                        except:
                            if entry[-1] is not None:
                                error_que.put([entry[-1], 'Img_path - date_not_found_or_wrong_format'])
                                continue
                            else:
                                error_que.put([entry[0], 'Img_path - image_missing_on_NGC'])
                                continue
                        if abs((study_date - birthdate).days) < 280:
                            if entry[-1] is None:
                                error_que.put([entry[0], 'Img_path - image_missing_on_NGC'])
                            else:
                                ps1 = entry[6]
                                ps2 = entry[7]
                                model = str(entry[3]) + ' - ' + str(entry[4])
                                img_path = entry[-1]
                                img_paths.append([img_path, ps1, ps2])
    
            if len(img_paths) > 0:
                temp['img_paths'] = img_paths
                data_que.put([cpr_phair_child, temp])
            else:
                not_found_que.put([cpr_phair_mother, cpr_phair_child, 'no_imgs_for_child'])  
        if worker_done_que.qsize() > 0:
            worker_done_que.put(1)
            break
        
def data_put_worker(working_dir, csv_que, worker_done_que):
    f = open(working_dir + "Registers/data.csv")
    f_csv = csv.reader(f)
    _ = next(f_csv)
    for row in f_csv:
        csv_que.put(row)
        if csv_que.qsize() > 5000:
            time.sleep(1)
    worker_done_que.put(1)

processes = []
p = mp.Process(target=data_put_worker, args=(working_dir, csv_que, worker_done_que))
p.start()
processes.append(p)

num_workers = min(num_workers, mp.cpu_count()-4)

logger.info(f"Starting {num_workers} workers - " + str(datetime.now().strftime('%H:%M:%S')))

for i in range(num_workers):
    p = mp.Process(target=worker, args=(mother_cpr_i, child_cpr_i, i_date, variables_from_csv, i_studydate, variables_from_db, working_dir, csv_que, error_que, not_found_que, data_que, worker_done_que))
    p.start()
    processes.append(p)

not_found = []
errors = []
info = {}
invalid_counter = 0
n = 0
while True:
    print(data_que.qsize())
    if error_que.qsize() > 0:
        error = error_que.get()
        errors.append(error)
    if not_found_que.qsize() > 0:
        not_found_item = not_found_que.get()
        not_found.append(not_found_item)

        n += 1
        if n % 1000 == 0:
            logger.info(f"Completed {n} files - " + str(datetime.now().strftime('%H:%M:%S')))

    if data_que.qsize() > 0:
        data = data_que.get()  
        cpr_phair_child = data[0]
       
        if cpr_phair_child == 'INVALID':
            cpr_phair_child = 'CHILD_' + str(invalid_counter)     
            invalid_counter += 1
        
        info[cpr_phair_child] = data[1]

        n += 1
        if n % 1000 == 0:
            logger.info(f"Completed {n} files - " + str(datetime.now().strftime('%H:%M:%S')))

    if worker_done_que.qsize() == len(processes):
        break

with open(working_dir + 'preprocessing/missing.csv', 'w', newline='') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    wr.writerow(["cpr_phair_mother", "cpr_phair_child", "error"])
    for row in not_found:
        wr.writerow(row)

with open(working_dir + 'preprocessing/errors.csv', 'w', newline='') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    wr.writerow(["info", "error"])
    for row in errors:
        wr.writerow(row)

with open(working_dir + 'preprocessing/data.json', 'w') as file:
    json.dump(info, file)
    
img_cpr_link = {}

with open(working_dir + 'preprocessing/cervix_check.csv', 'w') as file:
    wr = csv.writer(file)
    wr.writerow(["filename"])
    for key in info.keys():
        for img_path in info[key]['img_paths']:
            path = img_path[0]
            wr.writerow(path)
            img_cpr_link[path] = key
        
with open(working_dir + 'preprocessing/img_cpr_link.json', 'w') as file:
    json.dump(img_cpr_link, file)
        

for p in processes:
    p.join()

















        


