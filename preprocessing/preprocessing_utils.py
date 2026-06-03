#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:58:16 2026

@author: jacob
"""

import polars as pl
import operator
import sqlite3
from collections import Counter
from tqdm import tqdm

def unique(df, column, value):
    if value is True:
        df = df.filter(pl.col(column).count().over(column) == 1)
    elif value is False:
        df = df.filter(pl.col(column).count().over(column) != 1)
    return df

def in_list(df, column, value):
    df = df.filter(pl.col(column).is_in(value))
    return df

OPS = {">": operator.gt,
       "<": operator.lt,
       ">=": operator.ge,
       "<=": operator.le,
       "==": operator.eq,
       "!=": operator.ne}

custom_OPS = {"unique": unique,
              "in": in_list}

def load_table(path, ignore_errors=False, has_header=True):
    if path.endswith(".csv"):
        return pl.read_csv(path, ignore_errors=ignore_errors, has_header=has_header, infer_schema=False)

    else:
        raise NotImplementedError(f"Unknown file type for path: {path}")

def filter_conditions(df, condition, filter_on, table, external=True):        
    if condition.operator in custom_OPS.keys():
        df_temp = custom_OPS[condition.operator](df, condition.column, condition.value)
        if external:
            df_temp = df_temp.with_columns(pl.col(condition.match_on).alias(filter_on))
    else:    
        if condition.operator in ['>', '<', '>=', '<=']:
            df_temp = df_temp.with_columns(pl.col(condition.column).cast(pl.Int64, strict=False))
            df_temp = df_temp.filter(pl.col(condition.column).is_not_null())
        df_temp = df_temp.filter(OPS[condition.operator](pl.col(condition.column), condition.value))
        if external:
            df_temp = df_temp.with_columns(pl.col(condition.match_on).alias(filter_on))

    if condition.condition is None:
        table = df_temp.select(filter_on)
    elif condition.condition == "or":
        table = pl.concat([table, df_temp.select(filter_on)])
    elif condition.condition == "and":
        table = table.join(df_temp.select(filter_on), on=filter_on, how="inner")

    return table

def filter_df_internal(df, criteria):
    table = None
    for condition in criteria.conditions:
        table = filter_conditions(df, condition, 'CPR_MOTHER', table, external=False)
    
    if criteria.action == 'include':
        df = df.join(table, on=criteria.filter_on, how='semi')
    elif criteria.action == 'exclude':
        df = df.join(table, on=criteria.filter_on, how='anti')
    
    return df
    
def filter_df_external(df, criteria):
    table = None
    for condition in criteria.conditions:
        df_temp = load_table(condition.table)
        table = filter_conditions(df_temp, condition, criteria.filter_on, table)
   
    if criteria.action == 'include':
        df = df.join(table, on=criteria.filter_on, how='semi')
    elif criteria.action == 'exclude':
        df = df.join(table, on=criteria.filter_on, how='anti')

    return df
        
def mark_df_external(df, criteria):
    table = None
    for condition in criteria.conditions:
        df_temp = load_table(condition.table)
        table = filter_conditions(df_temp, condition, criteria.filter_on, table)
           
    if criteria.mark is True:
        df = df.with_columns(pl.col(criteria.filter_on).is_in(table[criteria.filter_on]).alias(criteria.mark_name))
    elif criteria.mark is False:
        df = df.with_columns(~pl.col(criteria.filter_on).is_in(table[criteria.filter_on]).alias(criteria.mark_name))
    else:
        raise Exception(f"Mark crtieria {criteria.mark} not understood. Use False or True only")
    
    return df

def sqlite_extractor(cfg, cpr_mothers):
    
    conn = sqlite3.connect(cfg.paths.SQL_DB)
    cur = conn.cursor()
    
    cur.execute("CREATE TEMP TABLE tmp_hashes (phair_hash TEXT PRIMARY KEY)")
    cur.executemany("INSERT INTO tmp_hashes VALUES (?)", [(h,) for h in cpr_mothers])
    conn.commit()
    
    metadata_dicom_variables = cfg.metadata_dicom_variables

    dicom_select = ",\n".join(f"d.{column}" for column, _ in metadata_dicom_variables)

    type_map = {"str": pl.Utf8,
                "float": pl.Float64,
                "int": pl.Int64,
                "date": pl.Utf8,
                "bool": pl.Boolean}

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
