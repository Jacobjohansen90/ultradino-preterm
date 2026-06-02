#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:58:16 2026

@author: jacob
"""

import polars as pl
import operator

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


    

def merge_population_tables(cfg, ignore_errors=False):
    df = pl.DataFrame()
    for cfg_table in cfg.population.tables:
        table = load_table(cfg_table.table, ignore_errors=ignore_errors)
        table = table.select(list(cfg_table.columns.values()))
        table = table.rename({v: k for k, v in cfg_table.columns.items()})
        table = table.select(sorted(table.columns))
        df = df.vstack(table)
        
    for name, t in cfg.population.types.items():
        if t == 'int':
            df = df.with_columns(pl.col(name).cast(pl.Int64, strict=False))
        elif t == 'date':
            df = df.with_columns(pl.col(name).str.to_date("%Y-%m-%d", strict=False))
        else:
            raise NotImplementedError(f"Unknown type {t}")

    return df

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