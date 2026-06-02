#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:01:10 2026

@author: jacob
"""

from preprocessing_utils import filter_df_internal, filter_df_external, mark_df_external, load_table
import polars as pl


custom_funcs = {'filter_df_internal': filter_df_internal,
                'filter_df_external': filter_df_external,
                'mark_df_external': mark_df_external}

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

def apply_inclusion_exclusion(df, cfg):
    discards = {}
    conditioned = {}
    mothers = df['CPR_MOTHER'].unique()
    children = df['CPR_CHILD'].unique()
    
    for criteria in cfg.image_criteria:
        fn = custom_funcs[criteria.function]
        df = fn(df, criteria)
        mothers_new = df['CPR_MOTHER'].unique()
        children_new = df['CPR_CHILD'].unique()
        discards[criteria.name] = {'mothers_discarded': len(mothers)-len(mothers_new),
                                   'children_discarded': len(children)-len(children_new),
                                   'mothers_cpr': mothers.filter(~mothers.is_in(mothers_new)),
                                   'children_cpr': children.filter(~children.is_in(children_new))}
        mothers = mothers_new
        children = children_new
        
    for criteria in cfg.population_criteria:            
        fn = custom_funcs[criteria.function]
        df = fn(df, criteria)
        mothers_new = df['CPR_MOTHER'].unique()
        children_new = df['CPR_CHILD'].unique()
        discards[criteria.name] = {'mothers_discarded': len(mothers)-len(mothers_new),
                                   'children_discarded': len(children)-len(children_new),
                                   'mothers_cpr': mothers.filter(~mothers.is_in(mothers_new)),
                                   'children_cpr': children.filter(~children.is_in(children_new))}
        mothers = mothers_new
        children = children_new
    
    for criteria in cfg.conditional_criteria:
        fn = custom_funcs[criteria.function]
        df = fn(df, criteria)
        n_mothers = df.filter(pl.col(criteria.mark_name)).get_column("CPR_MOTHER").n_unique()
        n_children = df.filter(pl.col(criteria.mark_name)).get_column("CPR_CHILD").n_unique()
        cpr_mothers = df.filter(pl.col(criteria.mark_name)).select("CPR_MOTHER").unique()
        cpr_children = df.filter(pl.col(criteria.mark_name)).select("CPR_CHILD").unique()
        conditioned[criteria.name] = {'mothers_conditioned': n_mothers,
                                      'children_conditioned': n_children,
                                      'mothers_cpr': cpr_mothers,
                                      'childrens_cpr': cpr_children}
    
    return df, discards