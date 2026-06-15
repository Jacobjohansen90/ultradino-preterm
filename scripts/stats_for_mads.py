#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:43:18 2026

@author: jacob
"""

import polars as pl


path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/'

#Full pop
df = pl.read_csv(path + 'OnlyFirstPreg_June_v2/data_dump/population.csv')
df = df.unique(subset=['CPR_CHILD'])

print("Full population:")
print(f"Mothers: {df['CPR_MOTHER'].n_unique()}")
print(f"Children: {df['CPR_CHILD'].n_unique()}")

count = df.select((pl.col("GA") >= 37*7).sum()).item()
print(f"X >= 37: {count}")
count = df.select(((pl.col("GA") < 37*7) & (pl.col("GA") >= 34*7)).sum()).item()
print(f"37 > X >= 34: {count}")
count = df.select(((pl.col("GA") < 34*7) & (pl.col("GA") >= 32*7)).sum()).item()
print(f"34 > X >= 32: {count}")
count = df.select((pl.col("GA") < 32*7).sum()).item()
print(f"X > 32: {count}")

#Cervix pop
path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/'
df = pl.read_csv(path + 'AnyPreg_June_v2/data_dump/filtered_population.csv', ignore_errors=True)
df = df.unique(subset=['CPR_CHILD'])

print ("Cervix scan pop:")
print(f"Mothers: {df['CPR_MOTHER'].n_unique()}")
print(f"Children: {df['CPR_CHILD'].n_unique()}")
count = df.select((pl.col("GA") >= 37*7).sum()).item()
print(f"X >= 37: {count}")
count = df.select(((pl.col("GA") < 37*7) & (pl.col("GA") >= 34*7)).sum()).item()
print(f"37 > X >= 34: {count}")
count = df.select(((pl.col("GA") < 34*7) & (pl.col("GA") >= 32*7)).sum()).item()
print(f"34 > X >= 32: {count}")
count = df.select((pl.col("GA") < 32*7).sum()).item()
print(f"X > 32: {count}")



