#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:59:31 2026

@author: jacob
"""

import polars as pl

path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/'

df_train = pl.read_parquet(path + 'AnyPreg_v5/train.parquet')
df_test = pl.read_parquet(path + 'AnyPreg_v5/test.parquet')

df_cl = pl.read_parquet(path + 'misc/CL_2009-2025.parqquet')


