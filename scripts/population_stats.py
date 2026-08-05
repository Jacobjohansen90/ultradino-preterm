#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:43:18 2026

@author: jacob
"""

import polars as pl

path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/'

train_data = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/AnyPreg_June_v3/train.parquet'
test_data = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/OnlyFirstPreg_June_v3/test.parquet'

remove_on_GA = ['c-section', 'induced']
cutoffs = [32,34,37]

train_df = pl.read_parquet(train_data)
test_df = pl.read_parquet(test_data)

print('--Training--')

for GA in cutoffs:
    print(f"--{GA}--")
    df_all = train_df.filter((pl.col("GA") // 7 >= GA) | ((pl.col("GA") // 7 < GA) & pl.all_horizontal(~pl.col(remove_on_GA))))
    df_preterm = train_df.filter((pl.col("GA") // 7 < GA) & pl.all_horizontal(~pl.col(remove_on_GA)))
    #Population count
    print('Births:')
    print(f"Total births: {df_all['CPR_CHILD'].n_unique()}")
    print(f"Preterm births: {df_preterm['CPR_CHILD'].n_unique()}")
    print()
    #Age
    print('Age')
    print(f"Total Age (+/- SD): {round(df_all['AGE'].mean(), 2)} ({round(df_all['AGE'].std(), 2)})")
    print(f"Preterm Age (+/- SD): {round(df_preterm['AGE'].mean(), 2)} ({round(df_preterm['AGE'].std(), 2)})")
    print()
    #BMI
    print('BMI')
    print(f"Total BMI (+/- SD): {round(df_all['BMI'].mean(), 2)} ({round(df_all['BMI'].std(), 2)})")
    print(f"Preterm BMI (+/- SD): {round(df_preterm['BMI'].mean(), 2)} ({round(df_preterm['BMI'].std(), 2)})")
    print()
    #Scanners
    print('Scanners')
    print('Total:')
    t = len(df_all) // 100
    counts = df_all['manufacturer_model_name'].value_counts()
    included = counts.filter(pl.col("count") >= t)
    excluded = counts.filter(pl.col("count") < t)
    for row in included.iter_rows(named=True):
        print(f"\t{row['manufacturer_model_name']}: {row['count']}")
        
    print(f"\tother: {excluded['count'].sum()}")
    
    print('Preterm:')
    t = len(df_preterm) // 100
    counts = df_preterm['manufacturer_model_name'].value_counts()
    included = counts.filter(pl.col("count") >= t)
    excluded = counts.filter(pl.col("count") < t)
    for row in included.iter_rows(named=True):
        print(f"\t{row['manufacturer_model_name']}: {row['count']}")
        
    print(f"\tother: {excluded['count'].sum()}")
    
    print()
    
print()    
print('--Test--')

for GA in cutoffs:
    print(f"--{GA}--")
    df_all = test_df.filter((pl.col("GA") // 7 >= GA) | ((pl.col("GA") // 7 < GA) & pl.all_horizontal(~pl.col(remove_on_GA))))
    df_preterm = test_df.filter((pl.col("GA") // 7 < GA) & pl.all_horizontal(~pl.col(remove_on_GA)))
    #Population count
    print('Births:')
    print(f"Total births: {df_all['CPR_CHILD'].n_unique()}")
    print(f"Preterm births: {df_preterm['CPR_CHILD'].n_unique()}")
    print()
    #Age
    print('Age')
    print(f"Total Age (+/- SD): {round(df_all['AGE'].mean(), 2)} ({round(df_all['AGE'].std(), 2)})")
    print(f"Preterm Age (+/- SD): {round(df_preterm['AGE'].mean(), 2)} ({round(df_preterm['AGE'].std(), 2)})")
    print()
    #BMI
    print('BMI')
    print(f"Total BMI (+/- SD): {round(df_all['BMI'].mean(), 2)} ({round(df_all['BMI'].std(), 2)})")
    print(f"Preterm BMI (+/- SD): {round(df_preterm['BMI'].mean(), 2)} ({round(df_preterm['BMI'].std(), 2)})")
    print()
    #Scanners
    print('Scanners')
    print('Total:')
    t = len(df_all) // 100
    counts = df_all['manufacturer_model_name'].value_counts()
    included = counts.filter(pl.col("count") >= t)
    excluded = counts.filter(pl.col("count") < t)
    for row in included.iter_rows(named=True):
        print(f"\t{row['manufacturer_model_name']}: {row['count']}")
        
    print(f"\tother: {excluded['count'].sum()}")
    
    print('Preterm:')
    t = len(df_preterm) // 100
    counts = df_preterm['manufacturer_model_name'].value_counts()
    included = counts.filter(pl.col("count") >= t)
    excluded = counts.filter(pl.col("count") < t)
    for row in included.iter_rows(named=True):
        print(f"\t{row['manufacturer_model_name']}: {row['count']}")
        
    print(f"\tother: {excluded['count'].sum()}")
    
    print()
    
    





    