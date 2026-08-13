#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:08:36 2026

@author: jacob
"""
import sqlite3
import numpy as np
import polars as pl
from concurrent.futures import ProcessPoolExecutor

SQL_path = '/projects/users/data/UCPH/DeepFetal/ultrasound/tables/DeepFetal_image_database_250526.sqlite'
save_path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/misc/cervix_preds_v3.csv'


def get_class(item):
    file_path, segmentation_path = item

    try:
        cls_logits = np.load(segmentation_path)["cls_logits"]
        cls = int(np.argmax(cls_logits))
        return file_path, cls
    except Exception as e:
        print(f"Failed: {segmentation_path}: {e}")
        return file_path, None


conn = sqlite3.connect(SQL_path)

paths = pl.read_database(
    """
    SELECT file_path, segmentation_path
    FROM path_table
    WHERE segmentation_path IS NOT NULL
    """,
    conn,
)

conn.close()

items = paths.iter_rows()

with ProcessPoolExecutor(max_workers=64) as executor:
    results = list(
        executor.map(
            get_class,
            items,
            chunksize=1000,
        )
    )

df = pl.DataFrame(
    results,
    schema=["file_path", "is_cervix"])

df.write_csv(save_path)