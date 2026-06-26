#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:28:32 2026

@author: jacob
"""
#%%Imports
import polars as pl
import operator

#%%Operator functions
def unique(df, column, value):
    if value is True:
        df = df.filter(pl.col(column).count().over(column) == 1)
    elif value is False:
        df = df.filter(pl.col(column).count().over(column) != 1)
    return df

def in_list(df, column, value):
    df = df.filter(pl.col(column).is_in(value))
    return df

def starts_with(df, column, value):
    df = df.filter(pl.col(column).str.starts_with(value))
    return df

def is_null(df, column, value):
    df = df.filter(pl.col(column).is_null())
    return df

OPS = {">": operator.gt,
       "<": operator.lt,
       ">=": operator.ge,
       "<=": operator.le,
       "==": operator.eq,
       "!=": operator.ne,
       '-': operator.sub,
       '+': operator.add}

custom_OPS = {"unique": unique,
              "in": in_list,
              "starts_with": starts_with,
              "is_null": is_null}

type_map = {"str": pl.Utf8,
            "float": pl.Float64,
            "int": pl.Int64,
            "date": pl.Utf8,
            "bool": pl.Boolean}

#%%Utility function
def load_table(path, ignore_errors=False, has_header=True):
    if path.endswith(".csv"):
        return pl.read_csv(path, ignore_errors=ignore_errors, has_header=has_header, infer_schema=False)

    else:
        raise NotImplementedError(f"Unknown file type for path: {path}")

def filter_conditions(df, condition, filter_on, table, action, external=True):      
    if condition.operator in custom_OPS.keys():
        df_temp = custom_OPS[condition.operator](df, condition.column, condition.value)
    else:    
        if condition.operator in ['>', '<', '>=', '<=', '-', '+']:
            df_temp = df.with_columns(pl.col(condition.column).cast(pl.Int64, strict=False))
            df_temp = df_temp.filter(pl.col(condition.column).is_not_null())
        else:
            df_temp = df
        df_temp = df_temp.filter(OPS[condition.operator](pl.col(condition.column), condition.value))

    if external:
        df_temp = df_temp.with_columns(pl.col(condition.match_on).alias(filter_on))
        if action == 'exclude_birth':
            filter_on = [filter_on, "cond_col"]
            df_temp = df_temp.with_columns(pl.col(condition.conditional_column)
                                           .str.strptime(pl.Date).alias('cond_col'))
            
    if condition.condition is None:
        table = df_temp.select(filter_on)
    elif condition.condition == "or":
        table = pl.concat([table, df_temp.select(filter_on)]).unique()
    elif condition.condition == "and":
        table = table.join(df_temp.select(filter_on), on=filter_on, how="semi")
    return table

def filter_df_internal(df, criteria):
    table = None
    for condition in criteria.conditions:
        table = filter_conditions(df, condition, criteria.filter_on, table, criteria.action, external=False)
    if criteria.action == 'include':
        df = df.join(table, on=criteria.filter_on, how='semi')
    elif criteria.action == 'exclude':
        df = df.join(table, on=criteria.filter_on, how='anti')
    return df
    
def filter_df_external(df, criteria):
    table = None
    for condition in criteria.conditions:
        df_temp = load_table(condition.table)
        table = filter_conditions(df_temp, condition, criteria.filter_on, table, criteria.action)
    if criteria.action == 'include':
        df = df.join(table, on=criteria.filter_on, how='semi')
    elif criteria.action == 'exclude':
        df = df.join(table, on=criteria.filter_on, how='anti')
    elif criteria.action == 'exclude_birth':
        matches = (df.join(table, on=criteria.filter_on, how="left")
                   .filter((pl.col("cond_col") <= pl.col("BIRTHDAY")) &
                           (pl.col("cond_col") >= pl.col("BIRTHDAY") - pl.duration(days=280)))
                   .select([criteria.filter_on, "BIRTHDAY"]))
        
        df = df.join(matches, on=[criteria.filter_on, "BIRTHDAY"], how="anti") 
        
    return df
        
def mark_df_external(df, criteria):
    table = None
    for condition in criteria.conditions:
        df_temp = load_table(condition.table)
        table = filter_conditions(df_temp, condition, criteria.filter_on, table, criteria.action)
           
    if criteria.action == 'include':
        df = df.with_columns(pl.col(criteria.filter_on).is_in(table[criteria.filter_on]).alias(criteria.mark_name))
    elif criteria.action == 'exclude':
        df = df.with_columns(~pl.col(criteria.filter_on).is_in(table[criteria.filter_on]).alias(criteria.mark_name))
    elif criteria.action == 'exclude_birth':
        matches = (df.join(table, on=criteria.filter_on, how="left")
                   .filter((pl.col("cond_col") <= pl.col("BIRTHDAY")) &
                           (pl.col("cond_col") >= pl.col("BIRTHDAY") - pl.duration(days=280)))
                   .select([criteria.filter_on, "BIRTHDAY"]))

        df = df.with_columns(pl.col(criteria.filter_on).is_in(matches[criteria.filter_on]).alias(criteria.mark_name))
        
        
    return df

def find_close_births(df, criteria):
    #Reduce to birth level
    births = (df.select(["CPR_MOTHER", "CPR_CHILD", criteria.column])
              .unique().sort(["CPR_MOTHER", criteria.column]))

    #Compute inter-mother birth gaps
    births = births.with_columns((pl.col(criteria.column).diff()
                                  .over("CPR_MOTHER").dt.total_days()
                                  .abs() < criteria.threshold).alias("close_births"))

    #Identify births that are close
    close_births = (births.filter(pl.col("close_births"))
                    .select(["CPR_MOTHER", "CPR_CHILD"]).unique())

    if criteria.action == 'include':
        df = df.join(close_births, on=["CPR_MOTHER", "CPR_CHILD"], how="semi")
    if criteria.action == 'exclude':
        df = df.join(close_births, on=["CPR_MOTHER", "CPR_CHILD"], how="anti")

    return df

def find_close_values(df, criteria):
    #Important! The first value in the filter_on list is assumed to be the one we sort over. 
    close_values = None
    for condition in criteria.conditions:
        df_temp = (df.select(criteria.filter_on + [condition.column])
                  .unique().sort([criteria.filter_on[0], criteria.column]))


        df_temp = df_temp.with_columns((pl.col(criteria.column).diff()
                                        .over(criteria.filter_on[0]).dt.total_days()
                                        .abs() < criteria.threshold).alias("close_values"))

        close_values = (df_temp.filter(pl.col("close_values"))
                        .select(criteria.filter_on).unique())

    if criteria.action == 'include':
        df = df.join(close_values, on=criteria.filter_on, how="semi")
    if criteria.action == 'exclude':
        df = df.join(close_values, on=criteria.filter_on, how="anti")

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
                                   'mothers_cpr': [mothers.filter(~mothers.is_in(mothers_temp)).to_list(), mothers_cpr],
                                   'children_cpr': [children.filter(~children.is_in(children_temp)).to_list(), children_cpr]}
        
        
    else:
        mothers_temp = df['CPR_MOTHER'].unique()
        children_temp = df['CPR_CHILD'].unique()
        discards[criteria.name] = {'mothers_discarded': len(mothers)-len(mothers_temp),
                                   'children_discarded': len(children)-len(children_temp),
                                   'mothers_cpr': mothers.filter(~mothers.is_in(mothers_temp)).to_list(),
                                   'children_cpr': children.filter(~children.is_in(children_temp)).to_list()}
    
    return discards, mothers_temp, children_temp


def condition(conditioned, df, criteria):    
    n_mothers = df.filter(pl.col(criteria.mark_name)).get_column("CPR_MOTHER").n_unique()
    n_children = df.filter(pl.col(criteria.mark_name)).get_column("CPR_CHILD").n_unique()
    cpr_mothers = df.filter(pl.col(criteria.mark_name))["CPR_MOTHER"].unique().to_list()
    cpr_children = df.filter(pl.col(criteria.mark_name))["CPR_CHILD"].unique().to_list()
        
        
    conditioned[criteria.name] = {'mothers_conditioned': n_mothers,
                                  'children_conditioned': n_children,
                                  'mothers_cpr': cpr_mothers,
                                  'childrens_cpr': cpr_children}
    
    return conditioned