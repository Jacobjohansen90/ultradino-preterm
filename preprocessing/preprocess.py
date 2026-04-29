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
import polars as pl

from workers import csv_extracter, db_crawler
from calc_stats import calc_stats
from EHR_extract.extract import merge_population_tables, extract_from_cfg, make_train_test_split
from utils.utils import unpack_dict_to_DF, pack_df_to_dict, match_images_with_child
#%%Load variable YAML and setup logger and dirs
cfg = OmegaConf.load('./confs/Preprocessing.yaml')

#Setup logger
logging.basicConfig(filename=cfg.paths.data_dir + 'preprocess.log', filemode='w', level=logging.INFO)
logger = logging.getLogger(__name__)

#Setup dirs
Path(cfg.paths.data_dir + 'logs/').mkdir(exist_ok=True)
Path(cfg.paths.data_dir + 'data_dump/').mkdir(parents=True, exist_ok=True)

#%%Build population data
cfg_population = OmegaConf.load(cfg.paths.population_yaml)
cfg_population.paths.data_dir = cfg.paths.data_dir

population = merge_population_tables(cfg_population.population.tables)

if cfg.debug:
    population = population[:1000]

population.write_csv(cfg_population.paths.data_dir + 'data_dump/population.csv')

n_births = mp.Value('i', population['CPR_MOTHER'].n_unique())

logger.info(f"Found {n_births.value} births - " + str(datetime.now().strftime('%H:%M:%S')))
    

#%%Crawl database
if not cfg.crawl_db:
    logger.info("Using existing database - " + str(datetime.now().strftime('%H:%M:%S')))
    df_img = pl.read_csv(cfg.paths.data_dir + 'data_dump/img_data.csv', ignore_errors=True)

else:
    #Setup ques, loggers and start processes
    in_que = mp.Queue()
    out_que = mp.Queue()
    done = mp.Value('b', False)
    db_idx = {}
    
    #Find population indexes
    variables = population.columns
    if not 'CPR_MOTHER' in variables:
        raise Exception("CPR_MOTHER must be present in population variables")

    
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
        p = mp.Process(target=db_crawler, args=(db_idx, cfg.paths.SQL_DB, 
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
        elif data[0] == 'CPR_error':
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
       
    with open(cfg.paths.data_dir + 'logs/cpr_errors.csv', 'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerow(['Error', 'CPR_MOTHER'])
        wr.writerows(not_found)
    
    df_img = unpack_dict_to_DF(final_data, 'imgs')
    
    #img_cpr_link = dict(zip(df['file_path'], df['CPR_CHILD']))
    
    # with open(cfg.paths.data_dir + 'data_dump/img_cpr_link.json', 'w') as file:
        # json.dump(img_cpr_link, file)
        
    df_img.write_csv(cfg.paths.data_dir + 'data_dump/img_data.csv')
    
    del not_found
    del errors_db
    del errors_img
 
#%%Apply inclusion/exclusion criteria

#Merge the img df with the EHR df
df = df_img.join(population, on='CPR_MOTHER', how='inner')

#Convert date columns to dates and link children and images
df = match_images_with_child(df, cfg_population.imaging_matching_criteria[0].args)


logging.info(f"Valid images: {len(df)} after matching image + EHR matching.  \n")

final_population, all_discards = extract_from_cfg(cfg_population, df)

discards = {}
for i in range(len(all_discards)):
    discards[i] = {"criteria": all_discards[i][0],
                   "n_discards": all_discards[i][2],
                   "n_population_pre_discard": all_discards[i][3],
                   "n_population_post_discard": all_discards[i][3] - all_discards[i][2],
                   "discards": all_discards[i][1]}

with open(cfg.paths.data_dir + 'logs/discards.json', "w") as file:
    json.dump(discards, file)

final_population.write_csv(cfg.paths.data_dir + 'data_dump/final_population.csv')

train_pop, test_pop = make_train_test_split(cfg.paths.holdout_csv, 
                                            final_population, 
                                            'file_path',
                                            cfg.SQL_prefix)

train_pop.write_csv(cfg.paths.data_dir + 'train.csv')
test_pop.write_csv(cfg.paths.data_dir + 'test.csv')

population_columns = list(cfg_population.population.tables[0]['columns'].keys())

train_pop_dict = pack_df_to_dict(train_pop, population_columns, cfg_population.population.population_key)
test_pop_dict = pack_df_to_dict(test_pop, population_columns, cfg_population.population.population_key)

with open(cfg.paths.data_dir + 'train.json', "w") as file:
    json.dump(train_pop_dict, file)

with open(cfg.paths.data_dir + 'test.json', "w") as file:
    json.dump(test_pop_dict, file)


"""
#%% Calculate stats
logger.info("Calculating stats - " + str(datetime.now().strftime('%H:%M:%S')))
calc_stats('/'.join(path.split('/')[:-2]) + '/')

logger.info("Preprocessing done - " + str(datetime.now().strftime('%H:%M:%S')))
"""