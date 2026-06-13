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
count = df.select((pl.col("GA") >= 37).sum()).item()
print(f"X >= 37: {count}")
count = df.select((pl.col("GA") < 37) & (pl.col("GA") >= 34).sum()).item()
print(f"37 > X >= 34: {count}")
count = df.select((pl.col("GA") < 34) & (pl.col("GA") >= 32).sum()).item()
print(f"34 > X >= 32: {count}")
count = df.select((pl.col("GA") > 32).sum()).item()
print(f"X > 32: {count}")

#Cervix pop
path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/'
df = pl.read_csv(path + 'AnyCervix_June_v2/data_dump/population.csv')
df = df.unique(subset=['CPR_CHILD'])

print ("Cervix scan pop:")
count = df.select((pl.col("GA") >= 37).sum()).item()
print(f"X >= 37: {count}")
count = df.select((pl.col("GA") < 37) & (pl.col("GA") >= 34).sum()).item()
print(f"37 > X >= 34: {count}")
count = df.select((pl.col("GA") < 34) & (pl.col("GA") >= 32).sum()).item()
print(f"34 > X >= 32: {count}")
count = df.select((pl.col("GA") > 32).sum()).item()
print(f"X > 32: {count}")

#Incl/Excl pop
path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/'
df = pl.read_csv(path + 'OnlyFirstPreg_June_v2/data_dump/population.csv')
df = df.unique(subset=['CPR_CHILD'])

print ("Incl/Excl pop:")
count = df.select((pl.col("GA") >= 37).sum()).item()
print(f"X >= 37: {count}")
count = df.select((pl.col("GA") < 37) & (pl.col("GA") >= 34).sum()).item()
print(f"37 > X >= 34: {count}")
count = df.select((pl.col("GA") < 34) & (pl.col("GA") >= 32).sum()).item()
print(f"34 > X >= 32: {count}")
count = df.select((pl.col("GA") > 32).sum()).item()
print(f"X > 32: {count}")


