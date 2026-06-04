#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:01:10 2026

@author: jacob
"""

import polars as pl
import sqlite3
from utils.preprocessing_utils import (filter_df_internal, 
                                       filter_df_external, 
                                       mark_df_external, 
                                       find_close_births,
                                       load_table, 
                                       OPS, 
                                       type_map,
                                       discard,
                                       condition)


custom_funcs = {'filter_df_internal': filter_df_internal,
                'filter_df_external': filter_df_external,
                'mark_df_external': mark_df_external,
                'find_close_births': find_close_births}


def link_t_tables(cfg):
    t_adm = pl.read_csv(cfg.t_tables.adm_table, infer_schema=False)
    for table in cfg.t_tables.tables:
        t_table = pl.read_csv(table.table, infer_schema=False)
        t_table = t_table.join(t_adm.select(table.include + [cfg.t_tables.link]), left_on=table.table_link, right_on=cfg.t_tables.link)
        t_table.write_csv(cfg.paths.data_dir + 'tables/' + table.table.split('/')[-1])
    

def merge_population_tables(cfg, ignore_errors=False):
    df = pl.DataFrame()
    for cfg_table in cfg.population.tables:
        table = load_table(cfg_table.table, ignore_errors=ignore_errors)
        table = table.select(list(cfg_table.columns.values()))
        table = table.rename({v: k for k, v in cfg_table.columns.items()})
        table = table.select(sorted(table.columns))
        df = df.vstack(table)
        
    for name, t in cfg.population.types.items():
        df = df.with_columns(pl.col(name).cast(type_map[t], strict=False))
        if t == 'date':
            df = df.with_columns(pl.col(name).str.strptime(pl.Date, strict=False))

    return df


def merge_population_and_image_df(df_img, df_pop, cfg):
    df = df_img.join(df_pop, on=cfg.merge.population_key, how='left')
    for config in cfg.merge.create_variables:
        if config.var_type == "days":
            df = df.with_columns(OPS[config.operator](pl.col(config.column_1),
                                                      pl.col(config.column_2)).dt.total_days().alias(config.var_name))
        else:
            df = df.with_columns(OPS[config.operator](pl.col(config.column_1),
                                                      pl.col(config.column_2)).cast(type_map[config.var_type]).alias(config.var_name))

    return df

def make_train_test_split(df, cfg, cols_to_check=['CPR_MOTHER', 'CPR_CHILD', 'no_ocr_preprocessed_file_path']):
    
    #df_holdout = pl.read_csv(cfg.paths.holdout_csv, has_header=False)
    
    df_holdout = (df.select("CPR_MOTHER").unique().sample(fraction=0.15, shuffle=True, seed=42))
    
    df_train = df.join(df_holdout, right_on="column_1", left_on="CPR_MOTHER", how="anti")
    df_test = df.join(df_holdout, right_on="column_1", left_on="CPR_MOTHER", how="semi")
    
    for col in cols_to_check:
        overlap = (df_train.select(col).unique().join(df_test.select(col).unique(),
                                                      on=col,
                                                      how="inner")
                   .get_column(col).to_list())  
        
        if len(overlap) > 0:
            print(overlap)
            raise Exception(f"Overlap in train and test split on {col}")    
    
    return df_train, df_test

def apply_inclusion_exclusion(df, cfg):
    discards = {}
    conditioned = {}
    mothers = df['CPR_MOTHER'].unique()
    children = df['CPR_CHILD'].unique()
    print(df.shape)

    for criteria in cfg.image_criteria:
        print(criteria.name)
        fn = custom_funcs[criteria.function]
        df = fn(df, criteria)
        print(df.shape)
        discards, mothers, children = discard(discards, df, criteria, mothers, children)
        
    for criteria in cfg.population_criteria:       
        print(criteria.name)
        fn = custom_funcs[criteria.function]
        df = fn(df, criteria)
        print(df.shape)
        discards, mothers, children = discard(discards, df, criteria, mothers, children)

    
    for criteria in cfg.conditional_criteria:
        print(criteria.name)
        fn = custom_funcs[criteria.function]
        df = fn(df, criteria)
        print(df.shape)
        conditioned = condition(conditioned, df, criteria)

    
    return df, discards, conditioned

def sqlite_extractor(cfg, cpr_mothers):
    
    conn = sqlite3.connect(cfg.paths.SQL_DB)
    cur = conn.cursor()
    
    cur.execute("CREATE TEMP TABLE tmp_hashes (phair_hash TEXT PRIMARY KEY)")
    cur.executemany("INSERT INTO tmp_hashes VALUES (?)", [(h,) for h in cpr_mothers])
    conn.commit()
    
    metadata_dicom_variables = cfg.imaging.metadata_dicom_variables

    dicom_select = ",\n".join(f"d.{column}" for column, _ in metadata_dicom_variables)

    schema = [("CPR_MOTHER", pl.Utf8),
              ("file_path", pl.Utf8),
              ("no_ocr_preprocessed_file_path", pl.Utf8),
              *[(column, type_map[dtype]) for column, dtype in metadata_dicom_variables]]
    
    cur.execute(f"""
                SELECT
                    t.phair_hash,
                    pt.file_path,
                    pt.no_ocr_preprocessed_file_path,
                    {dicom_select}
                FROM tmp_hashes t
                LEFT JOIN cpr_hashes c
                    ON c.phair_hash = t.phair_hash
                LEFT JOIN path_table pt
                    ON pt.file_hash = c.xxhash
                LEFT JOIN dicom_metadata_table d
                    ON d.sop_instance_uid = pt.sop_instance_uid
                """)

    df = pl.DataFrame(cur.fetchall(),
                      schema=schema,
                      orient="row",
                      strict=False)

    date_cols = [col for col, dtype in metadata_dicom_variables if dtype == "date"]

    df = df.with_columns([pl.col(col).str.strptime(pl.Date, format="%Y%m%d", strict=False) for col in date_cols])

    df = df.drop_nulls(subset="file_path")

    conn.close()
  
    return df