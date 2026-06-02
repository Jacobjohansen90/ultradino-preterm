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

from preprocessing.workers import sqlite_extractor
from preprocessing.calc_stats import calc_stats
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
cfg_population = OmegaConf.load('./confs/Population.yaml')
cfg_population.paths.data_dir = cfg.paths.data_dir

df_pop = merge_population_tables(cfg_population)

if cfg.debug:
    df_pop = df_pop[:10000]

df_pop.write_csv(cfg_population.paths.data_dir + 'data_dump/population.csv')

logger.info(f"Found {df_pop['CPR_MOTHER'].n_unique()} mothers - " + str(datetime.now().strftime('%H:%M:%S')))
    

#%%Crawl database

df = sqlite_extractor(cfg, list(df_pop['CPR_MOTHER']))

df.write_csv(cfg.paths.data_dir + 'data_dump/img_data.csv')

"""
    
    #Dump data into files
    with open(cfg.paths.data_dir + 'logs/db_errors.csv', 'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerow(["Error", "DB Query"])
        wr.writerows(errors_db)

    with open(cfg.paths.data_dir + 'logs/img_errors.csv', 'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerow(["Error", "Image Path"])
        wr.writerows(errors_img)

    with open(cfg.paths.data_dir + 'logs/cpr_errors.csv', 'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerow(['Error', 'CPR_MOTHER'])
        wr.writerows(not_found)

    df_img = unpack_dict_to_DF(final_data, 'imgs')

    #Add cervix predictions to DF.
    #Convert study date to date format
    df_cervix_preds = pl.read_csv(cfg.paths.cervix_preds)
    df_img = df_img.join(df_cervix_preds, on='file_path', how='left')
    df_img = df_img.with_columns(pl.col("study_date").cast(pl.Utf8).str.to_date("%Y%m%d"))
    del df_cervix_preds
    

    with open(cfg.paths.data_dir + 'data_dump/img_data.json', 'w') as file:
        json.dump(pack_df_to_dict(df_img, [], "CPR_MOTHER"), file)   
       
    df_img.write_csv(cfg.paths.data_dir + 'data_dump/img_data.csv')
    
    del not_found
    del errors_db
    del errors_img
 
#%%Apply inclusion/exclusion criteria



#Merge the img df with the EHR df
df = df_img.join(df_pop, on='CPR_MOTHER', how='inner')

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
"""
#%% Calculate stats
logger.info("Calculating stats - " + str(datetime.now().strftime('%H:%M:%S')))
calc_stats('/'.join(path.split('/')[:-2]) + '/')

logger.info("Preprocessing done - " + str(datetime.now().strftime('%H:%M:%S')))
"""