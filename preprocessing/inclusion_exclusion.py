#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:01:10 2026

@author: jacob
"""

import polars as pl
from preprocessing_utils import (filter_df_internal, 
                                 filter_df_external, 
                                 mark_df_external, 
                                 find_close_births,
                                 load_table, 
                                 OPS, 
                                 type_map)


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

def discard(discards, df, criteria, mothers, children):
    if criteria.name in discards.keys():
        mothers_temp = df['CPR_MOTHER'].unique()
        children_temp = df['CPR_CHILD'].unique()
        mothers_discarded = discards[criteria.name]['mothers_discarded']
        children_discarded = discards[criteria.name]['children_discarded']
        mothers_cpr = discards[criteria.name]['mothers_cpr']
        children_cpr = discards[criteria.name]['children_cpr']
        
        discards[criteria.name] = {'mothers_discarded': mothers_discarded + len(mothers)-len(mothers_temp),
                                   'children_discarded': children_discarded + len(children)-len(children_temp),
                                   'mothers_cpr': pl.concat([mothers.filter(~mothers.is_in(mothers_temp)), mothers_cpr]),
                                   'children_cpr': pl.concat([children.filter(~children.is_in(children_temp)), children_cpr])}
        
        
    else:
        mothers_temp = df['CPR_MOTHER'].unique()
        children_temp = df['CPR_CHILD'].unique()
        discards[criteria.name] = {'mothers_discarded': len(mothers)-len(mothers_temp),
                                   'children_discarded': len(children)-len(children_temp),
                                   'mothers_cpr': mothers.filter(~mothers.is_in(mothers_temp)),
                                   'children_cpr': children.filter(~children.is_in(children_temp))}

    return discards, mothers_temp, children_temp


def condition(conditioned, df, criteria):
    n_mothers = df.filter(pl.col(criteria.mark_name)).get_column("CPR_MOTHER").n_unique()
    n_children = df.filter(pl.col(criteria.mark_name)).get_column("CPR_CHILD").n_unique()
    cpr_mothers = df.filter(pl.col(criteria.mark_name)).select("CPR_MOTHER").unique()
    cpr_children = df.filter(pl.col(criteria.mark_name)).select("CPR_CHILD").unique()
        
        
    conditioned[criteria.name] = {'mothers_conditioned': n_mothers,
                                  'children_conditioned': n_children,
                                  'mothers_cpr': cpr_mothers,
                                  'childrens_cpr': cpr_children}
    
    return conditioned


def apply_inclusion_exclusion(df, cfg):
    discards = {}
    conditioned = {}
    mothers = df['CPR_MOTHER'].unique()
    children = df['CPR_CHILD'].unique()
    
    for criteria in cfg.image_criteria:
        print(criteria.name)
        fn = custom_funcs[criteria.function]
        df = fn(df, criteria)
        discards, mothers, children = discard(discards, df, criteria, mothers, children)
        
    for criteria in cfg.population_criteria:       
        print(criteria.name)
        fn = custom_funcs[criteria.function]
        df = fn(df, criteria)
        discards, mothers, children = discard(discards, df, criteria, mothers, children)

    
    for criteria in cfg.conditional_criteria:
        print(criteria.name)
        fn = custom_funcs[criteria.function]
        df = fn(df, criteria)
        conditioned, mothers, children = condition(conditioned, df, criteria, mothers, children)

    
    return df, discards, conditioned