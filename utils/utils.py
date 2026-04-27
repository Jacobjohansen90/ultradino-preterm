#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:30:20 2026

@author: jacob
"""

import polars as pl

def unpack_dict_to_list(dict, dict_key):
    # This function unpacks the list under dict_key
    # and returns a list of entries instead of a dict
    result = []
    first_iter = True
    for key, subdict in dict.items():
        for item in subdict.get(dict_key):
            if first_iter:
                headers = [k for k in subdict.keys() if k != dict_key] + list(item.keys())
                result.append(headers)
                first_iter = False
            new_list = [v for k, v in subdict.items() if k != dict_key] + list(item.values())
            result.append(new_list)
    return result

def unpack_dict_to_DF(dict, dict_key):
    temp_list = unpack_dict_to_list(dict, dict_key)
    df = pl.DataFrame(temp_list[1:], schema=temp_list[0], orient='row')
    return df

def pack_df_to_dict(df, meta_columns, population_key):
    img_cols = [c for c in df.columns if c not in meta_columns]
    grouped = (df.group_by(population_key).agg([pl.first(col).alias(col) for col in meta_columns if col != "CPR_CHILD"]
                                               +
                                               [pl.struct(img_cols).alias("imgs")]))
 
    final_dict =  {row[population_key]: {**{col: row[col] for col in meta_columns},
                                         "imgs": row["imgs"]} for row in grouped.to_dicts()}
    
    return final_dict



def pack_df_to_dict(df, EHR_columns, population_key):
    exprs = []
    for c, dtype in df.schema.items():
        if dtype == pl.Date:
            exprs.append(pl.col(c).dt.strftime("%Y-%m-%d").alias(c))
        elif dtype == pl.Datetime:
            exprs.append(pl.col(c).dt.strftime("%Y-%m-%d").alias(c))
        else:
            exprs.append(pl.col(c))

    df = df.with_columns(exprs)

    img_cols = [c for c in df.columns if c not in EHR_columns + [population_key]]

    grouped = (df.group_by(population_key).agg(
                [pl.first(col).alias(col) for col in EHR_columns]
                +
                [pl.struct(img_cols).alias("imgs")]))

    final_dict = {row[population_key]: {**{col: row[col] for col in EHR_columns},
                                        "imgs": row["imgs"]} for row in grouped.to_dicts()}

    return final_dict