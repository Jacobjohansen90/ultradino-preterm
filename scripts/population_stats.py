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

def print_scanner_counts(df, name, included_models):
    print(name)

    total = len(df)

    counts = (
        df["manufacturer_model_name"]
        .value_counts()
        .rename({"manufacturer_model_name": "model"})
    )

    other = 0

    for row in counts.iter_rows(named=True):
        if row["model"] in included_models:
            pct = 100 * row["count"] / total
            print(f"\t{row['model']}: {row['count']} ({pct:.1f}%)")
        else:
            other += row["count"]

    print(f"\tOther: {other} ({100 * other / total:.1f}%)")

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
    #Progesterone
    print('Progesterone')
    print(f"Total Progesterone (%): {df_all.unique('CPR_CHILD')['progesterone'].sum()} ({round(100 * df_all.unique('CPR_CHILD')['progesterone'].sum() / df_all['CPR_CHILD'].n_unique(), 2)}%)")    
    print(f"Preterm Progesterone (%): {df_preterm.unique('CPR_CHILD')['progesterone'].sum()} ({round(100 * df_preterm.unique('CPR_CHILD')['progesterone'].sum() / df_preterm['CPR_CHILD'].n_unique(), 2)}%)")    
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
    t = len(df_all) // 100

    counts_all = df_all["manufacturer_model_name"].value_counts()
    included_models = set(counts_all.filter(pl.col("count") >= t)["manufacturer_model_name"].to_list())    
    print_scanner_counts(df_all, "Total:", included_models)
    print_scanner_counts(df_preterm, "Preterm:", included_models)
    
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
    #Progesterone
    print('Progesterone')
    print(f"Total Progesterone (%): {df_all.unique('CPR_CHILD')['progesterone'].sum()} ({round(100 * df_all.unique('CPR_CHILD')['progesterone'].sum() / df_all['CPR_CHILD'].n_unique(), 2)}%)")    
    print(f"Preterm Progesterone (%): {df_preterm.unique('CPR_CHILD')['progesterone'].sum()} ({round(100 * df_preterm.unique('CPR_CHILD')['progesterone'].sum() / df_preterm['CPR_CHILD'].n_unique(), 2)}%)")    
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
    print_scanner_counts(df_all, "Total:", included_models)
    print_scanner_counts(df_preterm, "Preterm:", included_models)
    
    print()
    
    





    