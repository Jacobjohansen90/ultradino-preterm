#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:32:36 2026

@author: jacob
"""

from utils.test_utils import test_model

folder_path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Training_runs/Current/2026-06-24 07:15:11/'
test_data_path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/OnlyFirstPreg_June_v2/test.parquet'

test_model(folder_path, test_data_path)