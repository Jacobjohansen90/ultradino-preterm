#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:29:55 2026

@author: jj@di.ku.dk
"""
#%%Imports
import logging
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
import polars as pl
import json

from utils.calc_stats import calc_stats
from utils.utils import pack_df_to_dict
from utils.preprocessing_functions import (merge_population_tables, 
                                           merge_population_and_image_df, 
                                           apply_inclusion_exclusion, 
                                           link_t_tables,
                                           make_train_test_split,
                                           sqlite_extractor)
#%%Load variable YAML and setup logger and dirs
cfg = OmegaConf.load('./confs/Population.yaml')
cfg_incl_excl = OmegaConf.load(cfg.paths.incl_excl_cfg)
cfg_incl_excl.paths = cfg.paths
cfg.paths.data_dir += cfg_incl_excl.population_name + '/'

#Setup dirs
Path(cfg.paths.data_dir + 'data_dump/').mkdir(parents=True, exist_ok=False)
Path(cfg.paths.data_dir + 'logs/').mkdir()
Path(cfg.paths.data_dir + 'tables/').mkdir()

#Setup logger
logging.basicConfig(filename=cfg.paths.data_dir + 'logs/preprocess.log', filemode='w', level=logging.INFO)
logger = logging.getLogger(__name__)

#%%Build population data

df_pop = merge_population_tables(cfg)

link_t_tables(cfg)

if cfg.debug:
    df_pop = df_pop[:10000]

df_pop.write_csv(cfg.paths.data_dir + 'data_dump/population.csv')

logger.info(f"Found {df_pop['CPR_MOTHER'].n_unique()} mothers - " + str(datetime.now().strftime('%H:%M:%S')))
    

#%%Extract info from database

df_img = sqlite_extractor(cfg, list(df_pop['CPR_MOTHER'].unique()))

#Link cervix preds and image df
df_cervix_preds = pl.read_csv(cfg.paths.misc_dir + 'cervix_preds.csv', infer_schema=False)
df_img = df_img.join(df_cervix_preds, on='file_path', how='left')

df_img.write_csv(cfg.paths.data_dir + 'data_dump/img_data.csv')

del df_cervix_preds

#TODO: Remove this when DB is updated
#Currently using flow_imgs to detect multi images. Will be included in SQL DB at some point
df_flow = pl.read_csv(cfg.paths.misc_dir + 'flow_imgs.csv', infer_schema=False)
df_img = df_img.join(df_flow, left_on='file_path', right_on='filepath', how='anti')


logger.info(f"Found {len(df_img)} images - " + str(datetime.now().strftime('%H:%M:%S')))
logger.info(f"Found images for {df_img['CPR_MOTHER'].n_unique()} mothers - " + str(datetime.now().strftime('%H:%M:%S')))

#%%Merge image and population dfs
df = merge_population_and_image_df(df_img, df_pop, cfg)
df.write_csv(cfg.paths.data_dir + 'data_dump/test.csv')
#%%Apply inclusion/exclusion criteria

df, discards, conditioned = apply_inclusion_exclusion(df, cfg_incl_excl)

with open(cfg.paths.data_dir + 'logs/discards.json', "w") as file:
    json.dump(discards, file)

with open(cfg.paths.data_dir + 'logs/conditioned.json', "w") as file:
    json.dump(conditioned, file)
    
    
logger.info(f"Final data contains {len(df_img)} images - " + str(datetime.now().strftime('%H:%M:%S')))
logger.info(f"Final data contains {df_img['CPR_MOTHER'].n_unique()} mothers - " + str(datetime.now().strftime('%H:%M:%S')))
logger.info(f"Final data contains {df_img['CPR_CHILD'].n_unique()} children - " + str(datetime.now().strftime('%H:%M:%S')))

df.write_csv(cfg.paths.data_dir + 'data_dump/filtered_population.csv')

df_train, df_test = make_train_test_split(df, cfg)

df_train.write_csv(cfg.paths.data_dir + 'train.csv')
df_test.write_csv(cfg.paths.data_dir + 'test.csv')

population_columns = list(cfg.population.tables[0]['columns'].keys())

dict_train = pack_df_to_dict(df_train, population_columns, cfg.population.population_key)
dict_test = pack_df_to_dict(df_test, population_columns, cfg.population.population_key)

with open(cfg.paths.data_dir + 'train.json', "w") as file:
    json.dump(dict_train, file)

with open(cfg.paths.data_dir + 'test.json', "w") as file:
    json.dump(dict_test, file)

"""
#%% Calculate stats
logger.info("Calculating stats - " + str(datetime.now().strftime('%H:%M:%S')))
calc_stats('/'.join(path.split('/')[:-2]) + '/')

logger.info("Preprocessing done - " + str(datetime.now().strftime('%H:%M:%S')))
"""