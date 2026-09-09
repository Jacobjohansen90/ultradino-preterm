#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:28:32 2026

@author: jacob
"""
#%%Imports
import polars as pl
import operator
import sqlite3
from tqdm import tqdm
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import json
from scipy.spatial import ConvexHull

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)


#%%Operator functions

def unique(df, column, value):
    if value is True:
        df = df.filter(pl.len().over(column) == 1)
    elif value is False:
        df = df.filter(pl.len().over(column) != 1)
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
            "bool": pl.Boolean,
            "list": pl.List(pl.Float64)}

#%%Utility functions for inclusion / exclusion

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
        match_on = [condition.match_on] if isinstance(condition.match_on, str) else condition.match_on
        filter_on = [filter_on] if isinstance(filter_on, str) else filter_on
        df_temp = df_temp.with_columns(pl.col(src).alias(dst) for src, dst in zip(match_on, filter_on))
        if action in ['exclude_birth', 'include_birth']:
            filter_on = filter_on + ["date_of_occurence"]
            df_temp = df_temp.with_columns(pl.col(condition.date_column)
                                           .str.slice(0,10).strptime(pl.Date).alias('date_of_occurence'))
            
    if condition.condition is None:
        table = df_temp.select(filter_on)
    elif condition.condition == "or":
        table = pl.concat([table, df_temp.select(filter_on)]).unique()
    elif condition.condition == "and":
        table = table.join(df_temp.select(filter_on), on=filter_on, how="semi")
    return table

    
def filter_df(df, criteria):
    table = None
    for condition in criteria.conditions:
        print(condition)
        if 'table' in condition:
            df_temp = load_table(condition.table)
            table = filter_conditions(df_temp, condition, criteria.filter_on, table, criteria.action)
        else:
            table = filter_conditions(df, condition, criteria.filter_on, table, criteria.action, external=False)

    if criteria.action == 'include':
        df = df.join(table, on=criteria.filter_on, how='semi')
   
    elif criteria.action == 'exclude':
        df = df.join(table, on=criteria.filter_on, how='anti')
    
    elif criteria.action == 'include_birth':
        matches = (df.join(table, on=criteria.filter_on, how="left")
                   .filter((pl.col("date_of_occurence") <= pl.col("BIRTHDAY") + pl.duration(days=7)) &
                           (pl.col("date_of_occurence") >= pl.col("BIRTHDAY") - pl.duration(days=280)))
                   .select([criteria.filter_on, "BIRTHDAY"]))
        
        df = df.join(matches, on=[criteria.filter_on, "BIRTHDAY"], how="semi") 
    
    elif criteria.action == 'exclude_birth':
        matches = (df.join(table, on=criteria.filter_on, how="left")
                   .filter((pl.col("date_of_occurence") <= pl.col("BIRTHDAY") + pl.duration(days=7)) &
                           (pl.col("date_of_occurence") >= pl.col("BIRTHDAY") - pl.duration(days=280)))
                   .select([criteria.filter_on, "BIRTHDAY"]))
        
        df = df.join(matches, on=[criteria.filter_on, "BIRTHDAY"], how="anti") 
        
    return df
        
def mark_df_external(df, criteria):
    table = None
    for condition in criteria.conditions:
        df_temp = load_table(condition.table)
        table = filter_conditions(df_temp, condition, criteria.filter_on, table, criteria.action)
           
    if criteria.action == 'include':
        mark = pl.col(criteria.filter_on).is_in(table[criteria.filter_on])
        if criteria.mark_name in df.columns:
            df = df.with_columns((pl.col(criteria.mark_name) | mark).alias(criteria.mark_name))
        else:
            df = df.with_columns(mark.alias(criteria.mark_name))

    elif criteria.action == 'exclude':
        mark = ~pl.col(criteria.filter_on).is_in(table[criteria.filter_on])
        if criteria.mark_name in df.columns:
            df = df.with_columns((pl.col(criteria.mark_name) | mark).alias(criteria.mark_name))
        else:
            df = df.with_columns(mark.alias(criteria.mark_name))

    elif criteria.action == 'include_birth':
        mark = (df.join(table, on=criteria.filter_on, how="left")
                .filter((pl.col("date_of_occurence") <= pl.col("BIRTHDAY") + pl.duration(days=7)) &
                        (pl.col("date_of_occurence") >= pl.col("BIRTHDAY") - pl.duration(days=280)))
                .select([criteria.filter_on, "BIRTHDAY"])).unique().with_columns(pl.lit(True).alias('mark'))
        
        df = df.join(mark, on=[criteria.filter_on, "BIRTHDAY"], how="left") 

        if criteria.mark_name in df.columns:
            df = df.with_columns((pl.col(criteria.mark_name) | pl.col('mark').fill_null(False)).alias(criteria.mark_name))
        else:
            df = df.with_columns((pl.col('mark').fill_null(False)).alias(criteria.mark_name))
        
        df = df.drop('mark')

    
    elif criteria.action == 'exclude_birth':
        mark = (df.join(table, on=criteria.filter_on, how="left")
                .filter((pl.col("date_of_occurence") <= pl.col("BIRTHDAY") + pl.duration(days=7)) &
                        (pl.col("date_of_occurence") >= pl.col("BIRTHDAY") - pl.duration(days=280)))
                .select([criteria.filter_on, "BIRTHDAY"])).unique().with_columns(pl.lit(False).alias('mark'))

        df = df.join(mark, on=[criteria.filter_on, 'BIRTHDAY'], how='left')

        if criteria.mark_name in df.columns:
            df = df.with_columns(pl.when(pl.col("mark").is_not_null())
                                 .then(False).otherwise(pl.col(criteria.mark_name)).alias(criteria.mark_name))
        else:
            df = df.with_columns((pl.col('mark').fill_null(True)).alias(criteria.mark_name))
        
        df = df.drop('mark')

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

#%%High level inclusion / exclusion functions

custom_funcs = {'filter_df': filter_df,
                'mark_df_external': mark_df_external,
                'find_close_births': find_close_births}


def link_tables(cfg):
    for merge in cfg.merge_tables.merges:
        table = pl.read_csv(merge.table, infer_schema=False)
        merge_table = pl.read_csv(merge.merge_table, infer_schema=False)
        table = table.join(merge_table.select(merge.include + [merge.merge_table_link]), 
                           left_on=merge.table_link, right_on=merge.merge_table_link)
        
        table.write_csv(cfg.paths.data_dir + 'tables/' + merge.table.split('/')[-1])
    

def merge_population_tables(cfg, ignore_errors=False):
    df = pl.DataFrame()
    for cfg_table in cfg.population.tables:
        table = load_table(cfg_table.table, ignore_errors=ignore_errors)
        table = table.select(list(cfg_table.columns.values()))
        table = table.rename({v: k for k, v in cfg_table.columns.items()})
        table = table.select(sorted(table.columns))
        
        if df.height > 0:
            table = table.join(df.select(["CPR_MOTHER", "CPR_CHILD"]),
                               on=["CPR_MOTHER", "CPR_CHILD"],
                               how="anti")
            
        df = pl.concat([df, table])
        
    for name, t in cfg.population.types.items():
        df = df.with_columns(pl.col(name).cast(type_map[t], strict=False))
        if t == 'date':
            df = df.with_columns(pl.col(name).str.slice(0,10).str.strptime(pl.Date, strict=False))

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


def make_train_test_split(df, cfg, split):

    df_holdout = pl.read_csv(cfg.paths.holdout_csv)
    if split == 'test':        
        df = df.join(df_holdout, left_on="CPR_MOTHER", right_on="CPR_MOR", how="semi")
    elif split == 'train':
        df = df.join(df_holdout, left_on="CPR_MOTHER", right_on="CPR_MOR", how="anti")
    else:
        raise Exception(f"Split {split} not understood.")
    return df


def apply_inclusion_exclusion(df, cfg):
    discards = {}
    conditioned = {}
    mothers = df['CPR_MOTHER'].unique()
    children = df['CPR_CHILD'].unique()
    for criteria in cfg.image_criteria:
        fn = custom_funcs[criteria.function]
        df = fn(df, criteria)
        discards, mothers, children = discard(discards, df, criteria, mothers, children)

    for criteria in cfg.population_criteria:       
        fn = custom_funcs[criteria.function]
        df = fn(df, criteria)
        discards, mothers, children = discard(discards, df, criteria, mothers, children)

    
    for criteria in cfg.conditional_criteria:
        fn = custom_funcs[criteria.function]
        df = fn(df, criteria)
        conditioned = condition(conditioned, df, criteria)
    
    return df, discards, conditioned

#%%Cervical length functions

def get_CL(df, cfg):
    required_cols = ['physical_delta_x', 'physical_delta_y', 
                     'region_location_min_x0', 'region_location_max_x1',
                     'region_location_min_y0', 'region_location_max_y1']

    missing = set(required_cols) - set(df.columns)

    assert not missing, f"Missing columns: {missing}"
        
    with ProcessPoolExecutor(max_workers=64) as executor:
        results = list(tqdm(executor.map(calculate_CL,
                                         df.iter_rows(named=True),
                                         chunksize=1000),
                            total=df.height,
                            desc="Calculating CL",
                            unit="img"))
    
    df = df.with_columns(pl.Series("CL", results))
    
    return df

def calculate_CL(row, cervix_label=3):

    img_path = row['no_ocr_preprocessed_file_path']    
    seg_path = row['segmentation_path']    
    x0 = row['region_location_min_x0'][0]
    x1 = row['region_location_max_x1'][0]
    y0 = row['region_location_min_y0'][0] 
    y1 = row['region_location_max_y1'][0]
    delta_x = row['physical_delta_x'][0]
    delta_y = row['physical_delta_y'][0]
    
    if any(v is None for v in [x0, x1, y0, y1, delta_x, delta_y, img_path, seg_path]):
        return 0.0
        
    img = Image.open(img_path)
    seg = np.load(seg_path)['seg_logits']
    
    img_x, img_y = img.size
    seg_x, seg_y = seg.shape
    
    x_crop = x0 + (img_x - x1)
        
    ratio_x = (img_x - x_crop) / seg_x
    new_phys_delta_x = delta_x*ratio_x*10
    
    y_crop = y0 + (img_y - y1) 
    ratio_y = (img_y - y_crop) / seg_y
    new_phys_delta_y = delta_y*ratio_y*10
    
    xs, ys = np.where(seg == cervix_label)

    if len(xs) < 2:
        return 0.0
    
    if xs.max() - xs.min() <= ys.max() - ys.min():
        return 0.0

    coords = np.column_stack([xs * new_phys_delta_x,
                              ys * new_phys_delta_y])

    hull = ConvexHull(coords)
    hull_coords = coords[hull.vertices]

    diff = hull_coords[:, None, :] - hull_coords[None, :, :]
    CL = np.sqrt((diff ** 2).sum(axis=2)).max()
            
    return CL

#%%SQL functions
  
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
              ("segmentation_path", pl.Utf8),
              *[(column, type_map[dtype]) for column, dtype in metadata_dicom_variables],
              ("is_flow", pl.Boolean)]
    
    cur.execute(f"""
                SELECT
                    t.phair_hash,
                    pt.file_path,
                    pt.no_ocr_preprocessed_file_path,
                    pt.segmentation_path,
                    {dicom_select}
                FROM tmp_hashes t 
                LEFT JOIN cpr_hashes c
                    ON c.phair_hash = t.phair_hash
                LEFT JOIN path_table pt
                    ON pt.file_hash = c.xxhash
                LEFT JOIN dicom_metadata_table d
                    ON d.sop_instance_uid = pt.sop_instance_uid
                """)

    rows = []
    
    list_columns = {column for column, dtype in metadata_dicom_variables if dtype == "list"}
    
    for row in tqdm(cur.fetchall(), desc='Processing Rows'):
        row = list(row)

        is_flow = any(isinstance(s, str) and "[" in s for s in row)

        for i, (column, dtypes) in enumerate(metadata_dicom_variables, start=4):
            if column in list_columns:
                row[i] = to_list(row[i])
                
        rows.append((*row, is_flow))

    df = pl.DataFrame(rows,
                      schema=schema,
                      orient="row",
                      strict=False)


    df = df.drop_nulls(subset="file_path")

    date_cols = [col for col, dtype in metadata_dicom_variables if dtype == "date"]
    df = df.with_columns([pl.col(col).str.strptime(pl.Date, format="%Y%m%d", strict=False) for col in date_cols])

    conn.close()

    return df

def to_list(x):
    if x is None:
        return None

    if isinstance(x, str):
        if x.startswith("["):
            return [float(v) for v in json.loads(x)]
        return [float(x)]

    return [float(x)]