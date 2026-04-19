#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:29:55 2026

@author: jj@di.ku.dk
"""
#%%Imports
import csv
import json 
import logging
import multiprocessing as mp
from pathlib import Path
import sqlite3
from datetime import datetime
from omegaconf import OmegaConf

from workers import csv_extracter, db_crawler
from calc_stats import calc_stats
from EHR_extract.extract import merge_tables, inclusion_exclusion
from utils.utils import unpack_dict_to_DF
#%%Load variable YAML and setup logger and dirs
cfg = OmegaConf.load('./confs/Preprocessing.yaml')

#Setup logger
logging.basicConfig(filename=cfg.paths.data_dir + 'preprocess.log', filemode='w')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#Setup dirs
Path(cfg.paths.data_dir + 'logs/').mkdir(exist_ok=True)
Path(cfg.paths.data_dir + 'data_dump/').mkdir(parents=True, exist_ok=True)

#%%Build population data
cfg_population = OmegaConf.load(cfg.paths.population_yaml)

population = merge_tables(cfg_population)

if cfg.debug:
    population = population[:1000]

population.write_csv(cfg.paths.data_dir + 'data_dump/population.csv')

n_births = mp.Value('i', population.shape[0])

logger.info(f"Found {n_births.value} births - " + str(datetime.now().strftime('%H:%M:%S')))
    

#%%Crawl database
if not cfg.crawl_db:
    logger.info("Using existing database - " + str(datetime.now().strftime('%H:%M:%S')))
    with open(cfg.paths.data_dir + 'data_dump/img_data.json') as f:
        final_data = json.load(f)
    with open(cfg.paths.data_dir + 'data_dump/img_cpr_link.json') as f:
        img_cpr_link = json.load(f)
        
else:
    #Setup ques, loggers and start processes
    in_que = mp.Queue()
    out_que = mp.Queue()
    done = mp.Value('b', False)
    db_idx = {}
    pop_idx = {}
    
    #Find population indexes
    variables = population.columns
    if not 'CPR_CHILD' in variables and 'CPR_MOTHER' in variables:
        raise Exception("CPR_CHILD and CPR_MOTHER must be present in population variables")
    for i, variable in enumerate(variables):
        pop_idx[variable] = i
    
            
    #Crawl DB for variables indexes and check we found all of them
    with sqlite3.connect(cfg.paths.SQL_DB) as con:
        cur = con.cursor()
        cur.execute("SELECT * FROM metadata_cache LIMIT 0")
        db_headers = [desc[0] for desc in cur.description]
    
    for i in range(len(db_headers)):
        for variable in cfg.variables_from_db:
            if db_headers[i] == variable:
                db_idx[variable] = i
    if len(db_idx) != len(cfg.variables_from_db):
        found = list(db_idx.keys())
        diff = list(set(cfg.variables_from_db) - set(found))
        raise Exception(f"Did not find {diff} in database")
        
    num_workers = min(cfg.num_workers, mp.cpu_count()-4)
    
    logger.info(f"Starting {num_workers} workers - " + str(datetime.now().strftime('%H:%M:%S')))
    
    #Start crawler workers
    processes = []
    p = mp.Process(target=csv_extracter, args=(population, in_que, done))
    p.start()
    processes.append(p)
    
    for i in range(num_workers):
        p = mp.Process(target=db_crawler, args=(pop_idx, db_idx, cfg.paths.SQL_DB, 
                                                in_que, out_que, done))
        p.start()
        processes.append(p)
    
    not_found = []
    errors_db = []
    errors_img = []
    final_data = {}
    invalid_counter = 0
    n = 1
    
    while n <= n_births.value:
        data = out_que.get()
        if data[0] == 'DB_error':
            errors_db.append([data[1], data[2]])
        elif data[0] == 'img_error':
            errors_img.append([data[1], data[2]])
        elif data[0] == 'birth_not_found':
            n += 1
            not_found.append([data[1], data[2]])
            if n % 100000 == 0:
                logger.info(f"Completed {n} files - " + str(datetime.now().strftime('%H:%M:%S')))
        else:
            if data[0] == 'INVALID':
                final_data['INVALID_' + str(invalid_counter)] = data[1]
                invalid_counter += 1
            else:
                final_data[data[0]] = data[1]
            n += 1
            if n % 100000 == 0:
                logger.info(f"Completed {n} files - " + str(datetime.now().strftime('%H:%M:%S')))            
    
    #Shutdown processes
    for p in processes:
        p.terminate()
    
    #Dump data into files
    with open(cfg.paths.data_dir + 'logs/db_errors.csv', 'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerow(["Error", "DB Query"])
        wr.writerows(errors_db)

    with open(cfg.paths.data_dir + 'logs/img_errors.csv', 'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerow(["Error", "Image Path"])
        wr.writerows(errors_img)

    with open(cfg.paths.data_dir + 'data_dump/img_data.json', 'w') as file:
        json.dump(final_data, file)   
       
    with open(cfg.paths.data_dir + 'logs/birth_missing.csv', 'w', newline='') as file:
        wr = csv.writer(file)
        header = ['Error']
        for key in pop_idx.keys():
            header.append(key)
        wr.writerow(header)
        for row in not_found:
            wr.writerow([row[0]] + list(row[1].values()))
    
    df = unpack_dict_to_DF(final_data, 'imgs')
    img_cpr_link = dict(zip(df['CPR_CHILD'], df['file_path']))
    
    with open(cfg.paths.data_dir + 'data_dump/img_cpr_link.json', 'w') as file:
        json.dump(img_cpr_link, file)
        
    df.to_csv(cfg.paths.data_dir + 'data_dump/img_data.csv', index=False)
            
    del not_found
    del errors_db
    del errors_img
 
#%%Apply inclusion/exclusion criteria
cfg_incl_exl = OmegaConf.load(cfg.paths.incl_excl_yaml)

final_population, all_discards, img_metadata = inclusion_exclusion(population, cfg_incl_exl)



"""
#%%Make training and test data based on conditions
logger.info("Linking cervix preds with database - " + str(datetime.now().strftime('%H:%M:%S')))

f = open(path + 'image_data/misc/cervix_preds.csv')
d = csv.reader(f)
_ = next(d)

holdout_set = []

with open(holdout_path) as f:  
    holdout_csv = csv.reader(f)  

    for holdout_img in holdout_csv:
        holdout_set.append(holdout_img[0].split('PNG_processed_no_OCR/')[1])

holdout_set = set(holdout_set)

missing = []
cervix_data = {}
cervix_data_holdout = {}
cervix_data_SP = {}
cervix_data_SP_holdout = {}
excluded = []

img_not_in_db = []

for file in d:
    if file[1] == '14':
        if not os.path.isfile(path_imgs + file[0]):
            missing.append([path_imgs + file[0]])
        else:
            try:
                cpr_child = img_cpr_link[file[0]]
            except:
                img_not_in_db.append([file[0]])
                continue
            
            if final_data[cpr_child]['Induced'] == '1':
                excluded.append([cpr_child, 'Induced'])
            
            elif final_data[cpr_child]['C_section'] == '1' or 'KMCA' in final_data[cpr_child]['C_section']:
                excluded.append([cpr_child, 'C Section'])
            
            else:
                for imgs in final_data[cpr_child]['imgs']:
                    if imgs['file_path'] == file[0]:
                        img_data = imgs
                SP_date = datetime.strptime(img_data['study_date'], '%Y%m%d') >= SP_date_cutoff
                SP_reg = ('RegionH' in file[0]) or ('RegionSjaelland' in file[0])
                if file[0] in holdout_set:
                    if cpr_child in cervix_data_holdout.keys():
                        cervix_data_holdout[cpr_child]['imgs'].append(img_data)
                        if SP_date and SP_reg:
                            if cpr_child in cervix_data_SP_holdout.keys():
                                cervix_data_SP_holdout[cpr_child]['imgs'].append(img_data)
                            else:
                                cervix_data_SP_holdout[cpr_child] = {}
                                for key in final_data[cpr_child].keys():
                                    if key == 'imgs':
                                        cervix_data_SP_holdout[cpr_child][key] = [img_data]
                                    else:
                                        cervix_data_SP_holdout[cpr_child][key] = final_data[cpr_child][key]
        
                    else:
                        cervix_data_holdout[cpr_child] = {}
                        if SP_date and SP_reg:
                            cervix_data_SP_holdout[cpr_child] = {}
                            for key in final_data[cpr_child].keys():
                                if key == 'imgs':
                                    cervix_data_holdout[cpr_child][key] = [img_data]
                                    cervix_data_SP_holdout[cpr_child][key] = [img_data]
                                else:
                                    cervix_data_holdout[cpr_child][key] = final_data[cpr_child][key]
                                    cervix_data_SP_holdout[cpr_child][key] = final_data[cpr_child][key]
        
                        else:
                            for key in final_data[cpr_child].keys():
                                if key == 'imgs':
                                    cervix_data_holdout[cpr_child][key] = [img_data]
                                else:
                                    cervix_data_holdout[cpr_child][key] = final_data[cpr_child][key]
    
                else:            
                    if cpr_child in cervix_data.keys():
                        cervix_data[cpr_child]['imgs'].append(img_data)
                        if SP_date and SP_reg:
                            if cpr_child in cervix_data_SP.keys():
                                cervix_data_SP[cpr_child]['imgs'].append(img_data)
                            else:
                                cervix_data_SP[cpr_child] = {}
                                for key in final_data[cpr_child].keys():
                                    if key == 'imgs':
                                        cervix_data_SP[cpr_child][key] = [img_data]
                                    else:
                                        cervix_data_SP[cpr_child][key] = final_data[cpr_child][key]
        
                    else:
                        cervix_data[cpr_child] = {}
                        if SP_date and SP_reg:
                            cervix_data_SP[cpr_child] = {}
                            for key in final_data[cpr_child].keys():
                                if key == 'imgs':
                                    cervix_data[cpr_child][key] = [img_data]
                                    cervix_data_SP[cpr_child][key] = [img_data]
                                else:
                                    cervix_data[cpr_child][key] = final_data[cpr_child][key]
                                    cervix_data_SP[cpr_child][key] = final_data[cpr_child][key]
        
                        else:
                            for key in final_data[cpr_child].keys():
                                if key == 'imgs':
                                    cervix_data[cpr_child][key] = [img_data]
                                else:
                                    cervix_data[cpr_child][key] = final_data[cpr_child][key]


with open(path + 'traindata.json', 'w') as f:
    json.dump(cervix_data, f)

with open(path + 'traindata_SP.json', 'w') as f:
    json.dump(cervix_data_SP, f)

with open(path + 'testdata.json', 'w') as f:
    json.dump(cervix_data_holdout, f)

with open(path + 'testdata_SP.json', 'w') as f:
    json.dump(cervix_data_SP_holdout, f)

with open(path + 'logs/pred_img_missing.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['filepath_ngc'])
    writer.writerows(missing)
    
with open(path + 'logs/excluded.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['cpr_child', 'exclusion_criteria'])
    writer.writerows(excluded)
    

with open(path + 'logs/img_not_in_db.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['img_path'])
    writer.writerows(img_not_in_db)

#%% Calculate stats
logger.info("Calculating stats - " + str(datetime.now().strftime('%H:%M:%S')))
calc_stats('/'.join(path.split('/')[:-2]) + '/')

logger.info("Preprocessing done - " + str(datetime.now().strftime('%H:%M:%S')))
"""