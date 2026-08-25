#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:59:31 2026

@author: jacob
"""

import polars as pl
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.metrics import roc_auc_score

path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/'

df_full = pl.read_parquet(path + 'AnyPreg_v5/test.parquet')

for prog in [True, False]:
    print(f"-----Progesterone included: {prog}-----")
    if not prog:
        df_full = df_full.filter(~pl.col('progesterone'))
    for cutoff in [32,34,37]:
        print(f"\t-----GA {cutoff}-----")
        #Filter correct subsample
        df = df_full.filter(~((pl.col("GA") // 7 < cutoff) & (pl.col("induced") | pl.col("c-section"))))
        
        n_before = df.select(pl.col("CPR_CHILD").n_unique()).item()
        df = df.filter(pl.col("CL").max().over("CPR_CHILD") != 0)
        n_after = df.select(pl.col("CPR_CHILD").n_unique()).item()
        
        print(f"\tFraction removed due to CL = 0: {((n_before - n_after) / n_before):.4f}")
    
    
        #Do sens and spec at CL < 25
        df_child = (df.group_by("CPR_CHILD").agg([pl.col("GA").first().alias("GA"),
                                                  pl.col("CL").mean().alias("CL")]))
        
        y_true = (df_child["GA"] // 7 < cutoff).to_numpy()
        y_pred = (df_child["CL"] < 25).to_numpy()
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        sens_ci = proportion_confint(tp, tp + fn, alpha=0.05, method="wilson")
        spec_ci = proportion_confint(tn, tn + fp, alpha=0.05, method="wilson")
        spec_ci = proportion_confint(tn, tn + fp, alpha=0.05, method="wilson")
            
        print(f"\tSensitivity CL < 25: {sensitivity:.4f} (95% CI {sens_ci[0]:.4f}-{sens_ci[1]:.4f})")
        print(f"\tSpecificity CL < 25: {specificity:.4f} (95% CI {spec_ci[0]:.4f}-{spec_ci[1]:.4f})")
    
        #Do sens@85spec        
        cl = df_child["CL"].to_numpy()
        
        # Original estimate
        fpr, tpr, thresholds = roc_curve(y_true, -cl)
        specificity = 1 - fpr
        
        valid = np.where(specificity >= 0.85)[0]
        idx = valid[np.argmin(specificity[valid] - 0.85)]
        
        sens = tpr[idx]
        cl_cutoff = -thresholds[idx]
            
        # Bootstrap
        rng = np.random.default_rng()
        n_bootstrap = 2000
        
        boot_sens = []
        boot_cutoff = []
        
        for _ in range(n_bootstrap):
        
            # Resample children
            sample_idx = rng.integers(0, len(y_true), len(y_true))
        
            y_boot = y_true[sample_idx]
            cl_boot = cl[sample_idx]
        
            # Need both classes present
            if len(np.unique(y_boot)) < 2:
                continue
        
            fpr_b, tpr_b, thresholds_b = roc_curve(y_boot, -cl_boot)
            specificity_b = 1 - fpr_b
        
            valid_b = np.where(specificity_b >= 0.85)[0]
        
            if len(valid_b) == 0:
                continue
        
            idx_b = valid_b[np.argmin(specificity_b[valid_b] - 0.85)]
        
            boot_sens.append(tpr_b[idx_b])
            boot_cutoff.append(-thresholds_b[idx_b])
        
        # 95% bootstrap CIs
        sens_ci = np.percentile(boot_sens, [2.5, 97.5])
        cutoff_ci = np.percentile(boot_cutoff, [2.5, 97.5])
        
        print(f"Original cutoff: {cl_cutoff:.4f}")
        print(f"Bootstrap median: {np.median(boot_cutoff):.4f}")
        print(f"Bootstrap mean: {np.mean(boot_cutoff):.4f}")
        print(f"Bootstrap 2.5%: {np.percentile(boot_cutoff, 2.5):.4f}")
        print(f"Bootstrap 97.5%: {np.percentile(boot_cutoff, 97.5):.4f}")
        
        print(f"\tSensitivity @ 85% spec: "f"{sens:.4f} (95% CI {sens_ci[0]:.4f}-{sens_ci[1]:.4f})")    
        print(f"\tCL cutoff: "f"{cl_cutoff:.4f} (95% CI {cutoff_ci[0]:.4f}-{cutoff_ci[1]:.4f})")
        
        #Get AUC and CI
        auc = roc_auc_score(y_true, -cl)
        
        rng = np.random.default_rng()
        aucs = []
        
        for _ in range(2000):
            idx = rng.integers(0, len(y_true), len(y_true))
            aucs.append(roc_auc_score(y_true[idx], -cl[idx]))
        
        ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
        
        print(f"\tAUC: {auc:.4f} (95% CI {ci_low:.4f}-{ci_high:.4f})")