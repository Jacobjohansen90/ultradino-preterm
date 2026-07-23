#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:55:22 2026

@author: jacob
"""

import matplotlib
matplotlib.use('Agg')

import polars as pl
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from sklearn.metrics import auc, roc_curve
from scipy.stats import mannwhitneyu
import numpy as np
import re
from pathlib import Path
import logging
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def merge_dfs(pred_df, population_parquet):
    pred_df = pl.read_csv(pred_df)
    population_df = pl.read_parquet(population_parquet)
    df = pred_df.join(population_df, how='left', on='CPR_CHILD')
    
    return df

def prepare_columns(full_df, cfg):
    """Bin numeric factors into categorical groups."""
    
    #Deal with missing factors    
        # if factor not in df.columns:
            # print(f"  [skip] '{factor}' not found in dataframe")
            # continue
    
    if 'pred_max' in full_df.columns:
        pred = 'pred_max'
    else:
        pred = 'pred_avg'
    
    cols = ['CPR_CHILD', pred, 'label'] + [v.variable for v in cfg.variables]
    
    df = full_df.select(cols)
    
    df = df.rename({pred: 'pred'})
    
    for var_cfg in cfg.variables:
        var = var_cfg.variable
        if isinstance(var_cfg.bin, ListConfig):
            cutoffs = var_cfg.bin
            expr = (pl.when(pl.col(var) < cutoffs[0]).then(pl.lit(f"<{cutoffs[0]}")))

            for low, high in zip(cutoffs[:-1], cutoffs[1:]):
                expr = expr.when(pl.col(var) < high).then(pl.lit(f"{low}-{high}"))

            expr = expr.otherwise(pl.lit(f"{cutoffs[-1]}+"))

            df = df.with_columns(expr.alias(var))

    return df

def compute_global_performance(df, metric='sens@spec'):
    """Compute the global model performance on the full dataset"""
    sub = df[['label', 'pred']].drop_nulls()
    y_true, y_score = sub['label'], sub['pred']

    if y_true.n_unique() < 2:
        return np.nan

    if metric == 'aucroc':
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return float(auc(fpr, tpr))
    elif metric == 'sens@spec':
        return float(sensitivity_at_specificity_85(y_true, y_score))

def sensitivity_at_specificity_85(y_true, y_score, target_specificity=0.85):
    """Maximum sensitivity for a specificity close to 85% (tolerance grows if needed)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    specificity = 1 - fpr
    diffs = np.abs(specificity - target_specificity)
    tol = 0.005
    while not any(diffs <= tol):
        tol += 0.005
    return float(np.max(tpr[diffs <= tol]))

def extract_numeric_start(group_label):
    """Extract the first integer from a group label (e.g. '10 to 20 kg' -> 10)."""
    s = str(group_label)
    if s.startswith('<'):
        match = re.search(r'(\d+)', s)
        return (-1, int(match.group(1))) if match else (1000000, 0)
    elif s.startswith('>'):
        match = re.search(r'(\d+)', s)
        return (999999, int(match.group(1))) if match else (1000000, 0)
    match = re.match(r"(\d+)", s)
    if match:
        return (0, int(match.group(1)))
    return (1000000, 0)

def bootstrap_sensitivities(df, n_iterations=40):
    sensitivities = []
    for _ in range(n_iterations):
        sample = df.sample(n=len(df), replace=True)
        y_true = sample['label']
        y_score = sample['pred']
        if len(y_true.unique()) > 1:
            sensitivities.append(sensitivity_at_specificity_85(y_true, y_score))
    return sensitivities

def bootstrap_aucs(df, label_col, pred_col, n_iterations=40):
    """Bootstrap AUC-ROC values."""
    aucs = []
    for _ in range(n_iterations):
        sample = df.sample(n=len(df), replace=True)
        y_true = sample['label']
        y_score = sample['pred']
        if len(y_true.unique()) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            aucs.append(auc(fpr, tpr))
    return aucs

def compute_all_bias_metrics_classification(df, cfg):
    """Run compute_bias_per_factor_classification for all selected factors."""
    results = {}
    
    variables = [v.variable for v in cfg.variables]
    
    for variable in variables:
        result = compute_bias_per_variable_classification(df, variable, min_group_size=cfg.config.min_group_size, metric=cfg.metric)
        if result:
            results[variable] = result
            logger.info(f"[ok] {variable}: {len(result['subgroups'])} subgroups, "
                        f"{cfg.metric} {result['worst_value']:.2f} - {result['best_value']:.2f}")
        else:
            logger.info(f"[skip] {variable}: not enough valid subgroups")
    return results



def compute_bias_per_variable_classification(df, variable, min_group_size=100, metric='sens@spec'):
    """Compute classification metric + bootstrap for each subgroup of a factor."""
    df_temp = df.filter(pl.col(variable).is_not_null())
    group_counts = df_temp.group_by(variable).len()
    valid = (group_counts.filter(pl.col("len") >= min_group_size).select(variable).to_series())

    if len(valid) < 2:
        return None
    
    #TODO: This is kind of dirty, since we just let str cats be sorted randomly
    sorted_groups = sorted(valid, key=extract_numeric_start)

    res, boot, counts = {}, {}, {}
    for sg in sorted_groups:
        print(sg)
        sub = df_temp.filter(pl.col(variable) == sg)
        y_true = sub['label']
        y_score = sub['pred']
        if y_true.n_unique() < 2:
            logger.info(f"[skip] '{sg}': constant label")
            continue
        if metric == 'auc':
            fpr, tpr, _ = roc_curve(y_true, y_score)
            val = auc(fpr, tpr)
            boot_vals = bootstrap_aucs(sub, n_iterations=40)
        elif metric == 'sens@spec':
            val = sensitivity_at_specificity_85(y_true, y_score)
            boot_vals = bootstrap_sensitivities(sub, n_iterations=40)

        if not np.isnan(val):
            res[sg] = val
            boot[sg] = boot_vals
            counts[sg] = len(sub)

    if len(res) < 2:
        return None

    valid_sorted = [sg for sg in sorted_groups if sg in metric]
    best = max(res, key=res.get)
    worst = min(res, key=res.get)

    return {'subgroups': valid_sorted,
            'metric_per_subgroup': metric,
            'bootstrap': boot,
            'counts': counts,
            'best_subgroup': best,
            'worst_subgroup': worst,
            'best_value': res[best],
            'worst_value': res[worst]}


def _radar_polygon_area(vals, angles, ymin=0.0):
    """
    Compute the area of a closed radar polygon using the shoelace formula.

    The baseline (ymin) is subtracted so that only the visible portion of the
    polygon contributes to the area.  The last point in vals/angles must repeat
    the first (closed loop, as produced by angles + [angles[0]]).

    Parameters
    ----------
    vals   : array-like  – radial values (e.g. best_plot or worst_plot)
    angles : array-like  – corresponding angles in radians (closed loop)
    ymin   : float       – radar baseline (e.g. 0.0 for regression, 0.5 for AUC)

    Returns
    -------
    float : polygon area ≥ 0
    """
    r = np.maximum(np.array(vals, dtype=float) - ymin, 0.0)
    theta = np.array(angles, dtype=float)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))

def compute_mean_abs_gap(bias_results):
    """
    Compute the mean absolute gap between the worst and best subgroup metric,
    averaged over all factors in bias_results.

    For regression : gap = worst_value - best_value  (higher MRE is worse)
    For classification : gap = best_value - worst_value (higher sensitivity is better)
    In both cases the gap is abs(best_value - worst_value) >= 0.

    Returns
    -------
    float : mean |worst_value - best_value| across all factors
    """
    if not bias_results:
        return np.nan
    gaps = [abs(res['worst_value'] - res['best_value']) for res in bias_results.values()]
    return float(np.mean(gaps))

def compute_mean_rel_gap(bias_results):
    """
    Compute the mean relative gap between the worst and best subgroup metric,
    averaged over all factors in bias_results.

    rel_gap = |max - min| / max * 100  (%)
    where max = max(best_value, worst_value) for each factor.

    Returns
    -------
    float : mean relative gap (%) across all factors
    """
    if not bias_results:
        return np.nan
    rel_gaps = []
    for res in bias_results.values():
        denom = max(abs(res['best_value']), abs(res['worst_value']))
        if denom > 0:
            rel_gaps.append(abs(res['worst_value'] - res['best_value']) / denom * 100)
    return float(np.mean(rel_gaps)) if rel_gaps else np.nan

def _significance_test(values_a, values_b):
    """
    Return (p_value, stars) for a two-sample comparison.

    Regression  : Welch's t-test on raw MRE arrays (values_a / values_b are
                  1-D float arrays).
    Classification : Mann-Whitney U on bootstrap distributions (values_a /
                  values_b are 1-D float arrays of bootstrap metric values).

    Stars: '**' p<0.001 | '*' p<0.05 | '' otherwise.
    """
    if len(values_a) < 5 or len(values_b) < 5:
        return 1.0, ''
    try:
        _, p = mannwhitneyu(values_a, values_b, alternative='two-sided')
        stars = '**' if p < 0.001 else ('*' if p < 0.05 else '')
        return float(p), stars
    except Exception:
        return 1.0, ''

def _metric_label(metric='sensitivity'):
    if metric == 'auc':
        return 'AUC-ROC'
    elif metric == 'sens@spec':
        return 'Sens @ 85% Spec'

def _radar_ylim(bias_results=None, metric='sensitivity', ylim=None):
    """
    Return (ymin, ymax, yticks, yticklabels) for radar polar axes.

    Priority: explicit ylim > cls_metric-based defaults > data-driven (regression).
    """
    if ylim is not None:
        ymin, ymax = ylim
        step = 0.1 if (ymax - ymin) <= 0.6 else 0.2
        ticks = [round(t, 2) for t in np.arange(ymin + step, ymax, step)]
        return ymin, ymax, ticks, [f'{t:.1f}' for t in ticks]

    if metric == 'auc':
        ticks = [0.5, 0.6, 0.7, 0.8, 0.9]
        return 0.5, 1.0, ticks, ['0.5', '0.6', '0.7', '0.8', '0.9']
    elif metric == 'sens@spec':
        ticks = [0.2, 0.4, 0.6, 0.8]
        return 0.0, 1.0, ticks, ['0.2', '0.4', '0.6', '0.8']

def plot_radar_main(bias_results, global_perfomance, cfg):
    """
    Radar showing best/worst group for each factor.
    Works for both regression (MRE) and classification (sensitivity / AUC-ROC).

    Parameters
    ----------
    ylim : (ymin, ymax) | None
        Override the y-axis range. Use compute_shared_scale() to align multiple
        figures generated for different models on the same scale.
    cls_metric : str
        'sensitivity' or 'aucroc'  (ignored for regression).
    show_factor_name : bool
        If True (default), display the factor name in bold above the subgroup
        labels.  Set to False to only show the subgroup names (cleaner for
        publication figures with many factors).
    show_metric_value : bool
        If True, append the numeric metric value after each subgroup label
        (e.g. '↑Caucasian 5.23%' / '↓Asian 8.10%').  Default False.

    Quadrant support
    ----------------
    Set config['quadrants'] to a list of dicts to group factors into colored sectors:

        config['quadrants'] = [
            {'label': 'Mother',  'vars': ['Ethnicity', 'BMI', 'Mother Age'], 'color': '#d0e1f2'},
            {'label': 'Baby',    'vars': ['GA', 'Birth Weight', 'CL'],       'color': '#e6d6f0'},
            {'label': 'Imaging', 'vars': ['Device Name', 'Pixel Spacing'],   'color': '#fbe2e5'},
            {'label': 'Context', 'vars': ['Birth Year', 'Birth Place'],      'color': '#d9f2e6'},
        ]

    Each quadrant spans 360/n_quadrants degrees. Factors absent from bias_results are
    skipped. Factors not listed in any quadrant are appended after the last one.
    Without config['quadrants'], factors are spread evenly (previous behaviour).
    """
    if not bias_results:
        logger.warning("Warning: no bias results to plot")
        return None

    quadrant_def = None

    #TODO: Implement this - needs to come from CFG
    if quadrant_def:
        n_quads      = len(quadrant_def)
        deg_per_quad = 360.0 / n_quads
        variables      = []
        angle_map    = {}

        for q_idx, q in enumerate(quadrant_def):
            start_deg  = q_idx * deg_per_quad
            end_deg    = start_deg + deg_per_quad
            vars_in_q  = [v for v in q['vars'] if v in bias_results]
            if not vars_in_q:
                continue
            # Distribute inside (start, end) exclusive of boundary ticks
            theta_range = np.linspace(start_deg, end_deg, len(vars_in_q) + 2)[1:-1]
            for n_var, (var, angle_deg) in enumerate(zip(vars_in_q, theta_range)):
                # Nudge extreme variables outward toward the quadrant edges
                # (mirrors the adjustment in plot_bias_radar_by_quadrant)
                if len(vars_in_q) == 2:
                    if n_var == 0:
                        angle_deg -= deg_per_quad / 16
                    elif n_var == len(vars_in_q) - 1:
                        angle_deg += deg_per_quad / 20
                elif len(vars_in_q) == 3:
                    if n_var == 0:
                        angle_deg -= deg_per_quad / 20
                    elif n_var == len(vars_in_q) - 1:
                        angle_deg += deg_per_quad / 20
                angle_map[var] = np.deg2rad(angle_deg)
                variables.append(var)

        # Append factors not listed in any quadrant (evenly, after the last sector)
        unlisted = [f for f in bias_results if f not in angle_map]
        if unlisted:
            extra_angles = np.linspace(
                np.deg2rad(n_quads * deg_per_quad),
                np.deg2rad(n_quads * deg_per_quad + 360),
                len(unlisted) + 2
            )[1:-1]
            for var, ang in zip(unlisted, extra_angles):
                angle_map[var] = ang % (2 * np.pi)
                variables.append(var)
    else:
        variables = list(bias_results.keys())
        angles_eq = np.linspace(0, 2 * np.pi, len(variables), endpoint=False)
        angle_map = {f: a for f, a in zip(variables, angles_eq)}

    n = len(variables)
    if n < 2:
        print("Warning: fewer than 2 factors – radar cannot be drawn")
        return None

    ymin, ymax, yticks, yticklabels = _radar_ylim(bias_results=bias_results, metric=cfg.metric, ylim=cfg.config.ylim)

    angles = [angle_map[v] for v in variables]
    best_vals = [bias_results[v]['best_value']  for v in variables]
    worst_vals = [bias_results[v]['worst_value'] for v in variables]

    # Close the loop
    angles_plot = angles + [angles[0]]
    best_plot   = best_vals  + [best_vals[0]]
    worst_plot  = worst_vals + [worst_vals[0]]

    # ── Bias index ────────────────────────────────────────────────────────────
    # Areas are computed on the baseline-offset polygons (visible part only).
    # bias_index = area_inner / area_outer  (min / max of the two polygons)
    #   → 1.0 : no bias  (best ≈ worst)
    #   → 0.0 : maximum bias  (inner polygon collapses)
    # Works identically for regression (best=inner) and classification (worst=inner).
    area_best  = _radar_polygon_area(best_plot,  angles_plot, ymin)
    area_worst = _radar_polygon_area(worst_plot, angles_plot, ymin)
    area_outer = max(area_best, area_worst)
    bias_index = min(area_best, area_worst) / area_outer if area_outer > 0 else np.nan
    bias_label = f"Bias index (inner/outer) = {bias_index:.3f}"

    mean_gap = compute_mean_abs_gap(bias_results)
    mean_rel_gap = compute_mean_rel_gap(bias_results)
    mean_gap_label = (f"Mean |worst - best| = {mean_gap:.3f}"
                      f"  ({mean_rel_gap:.1f}%)")

    logger.info(f"  {bias_label}")
    logger.info(f"  {mean_gap_label}")

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, color='gray', size=cfg.config.font_size-8)

    # ── Quadrant background shading + separator lines ─────────────────────────
    #if quadrant_def:
    #    for q_idx, q in enumerate(quadrant_def):
    #        start_rad = np.deg2rad(q_idx * deg_per_quad)
    #        end_rad   = np.deg2rad((q_idx + 1) * deg_per_quad)
    #        ax.fill_between(np.linspace(start_rad, end_rad, 100),
    #                        ymin, ymax, color=q['color'], alpha=0.3)
    #    for q_idx in range(n_quads):
    #        sep = np.deg2rad(q_idx * deg_per_quad)
    #        ax.plot([sep, sep], [ymin, ymax * 1.05], color='black', linewidth=1.5)

    ax.vlines(angles_plot, ymin, ymax, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.plot(angles_plot, best_plot, color='blue', linewidth=2,
            label='Best Group', alpha=0.6)
    ax.plot(angles_plot, worst_plot, color='red', linewidth=2,
            label='Worst Group', alpha=0.6)
    ax.fill(angles_plot, best_plot, color='blue', alpha=0.1)
    ax.fill(angles_plot, worst_plot, color='red', alpha=0.1)

    ax.set_xticks([])
    
    for angle, variable in zip(angles, variables):
        res = bias_results[variable]
        best_sg  = res['best_subgroup']
        worst_sg = res['worst_subgroup']


        a = res['bootstrap'].get(best_sg,  np.array([]))
        b = res['bootstrap'].get(worst_sg, np.array([]))
        _, significance = _significance_test(a, b)

        # ── Metric value suffix ───────────────────────────────────────────
        if cfg.show_metric_value:
            bv = res['best_value']
            wv = res['worst_value']
            best_val_str  = f' {bv:.2f}'
            worst_val_str = f' {wv:.2f}'
        else:
            best_val_str = worst_val_str = ''

        label_best  = f'↑{best_sg}{significance}{best_val_str}'
        label_worst = f'↓{worst_sg}{worst_val_str}'

        children = []
        if cfg.show_factor_name:
            children.append(TextArea(variable, textprops=dict(size=cfg.plots.font_size + 3, weight='bold')))
        if cfg.show_subgroup_names:
            children.append(TextArea(label_best,  textprops=dict(size=cfg.plots.font_size - 2, color='blue')))
            children.append(TextArea(label_worst, textprops=dict(size=cfg.plots.font_size - 2, color='red')))

        vpack = VPacker(children=children, align="center", pad=0, sep=1)
        ab = AnnotationBbox(vpack, (angle, ymax),
                            frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)

    metric_lbl = _metric_label(cfg.metric)
    if global_perfomance is not None and not np.isnan(global_perfomance):
        fmt  = '.3f'
        global_str = f"Global {metric_lbl}: {global_perfomance:{fmt}}"
    else:
        global_str = ''
    
    
    if cfg.plots.show_title:
        ax.set_title(
            f"{metric_lbl}"
            + (f"\n{global_str}" if global_str else '')
            + f"\n{bias_label}"
            + f"\n{mean_gap_label}\n",
            fontsize=cfg.plots.font_size + 5, pad=30,
        )
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), fontsize=cfg.plots.font_size-8)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    out = Path(cfg.config.save_path)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / 'radar_main.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Main radar saved: {save_path}")
    return str(save_path), bias_index, mean_gap, mean_rel_gap

def plot_radar_comparison_per_factor(bias_results, variable, cfg):
    """import polars as pl

    Radar with one axis per subgroup for a given factor (single model).
    Displays the metric value of each subgroup.

    Parameters
    ----------
    ylim : (ymin, ymax) | None
        Override the y-axis range for cross-model comparability.
    cls_metric : str   'sensitivity' or 'aucroc'  (ignored for regression).
    """
    if variable not in bias_results:
        return None

    res = bias_results[variable]
    subgroups = res['subgroups']
    metric = res['metric_per_subgroup']
    counts = res['counts']
    n = len(subgroups)

    if n < 2:
        return None

    vals = [metric[sg] for sg in subgroups]

    # Scale – use _radar_ylim for consistency with other radars
    ymin, ymax, yticks, yticklabels = _radar_ylim(bias_results={variable: res}, metric=cfg.metric, ylim=cfg.plots.ylim)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_plot = angles + [angles[0]]
    vals_plot = vals + [vals[0]]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, color='gray', size=cfg.plots.font_size-8)
    ax.vlines(angles_plot, ymin, ymax, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.plot(angles_plot, vals_plot, color='teal', linewidth=2, alpha=0.6)
    ax.fill(angles_plot, vals_plot, color='teal', alpha=0.1)

    ax.set_xticks([])

    best_sg = res['best_subgroup']

    for angle, sg in zip(angles, subgroups):
        val = metric[sg]
        count = counts[sg]

        # Statistical test vs best subgroup
        stars = ''
        if sg != best_sg:
            a = res['bootstrap'].get(sg, np.array([]))
            b = res['bootstrap'].get(best_sg, np.array([]))
            _, stars = _significance_test(a, b)

        txt1 = TextArea(f'{sg}{stars}', textprops=dict(size=cfg.plots.font_size, weight='bold'))
        txt2 = TextArea(f'n={count}', textprops=dict(size=cfg.plots.font_size - 4, color='black'))
        txt3 = TextArea(f'{val:.2f}', textprops=dict(size=cfg.plots.font_size - 4, color='teal'))

        vpack = VPacker(children=[txt1, txt2, txt3], align="center", pad=0, sep=1)
        ab = AnnotationBbox(vpack, (angle, ymax),
                            frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)

    metric_lbl = _metric_label(cfg.metric)
    ax.set_title(f"{variable}\n{metric_lbl}", fontsize=30, pad=60)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    out = Path(cfg.config.save_path)
    name = variable.replace(' ', '_').replace('/', '_')
    save_path = out / f'radar_{name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Radar {variable}: {save_path}")
    return str(save_path)

def plot_all_comparison_radars(bias_results, cfg):
    paths = []
    for variable in bias_results:
        p = plot_radar_comparison_per_factor(bias_results, variable, cfg)
        if p:
            paths.append(p)
    return paths

def save_metrics_csv(bias_results, cfg, df=None):
    rows = []

    for variable, res in bias_results.items():
        best_sg = res['best_subgroup']
        for sg in res['subgroups']:
            val = res['metric_per_subgroup'][sg]

            # Statistical test vs best subgroup
            if sg == best_sg:
                p_value, significance = 1.0, ''
            else:
                a = res['bootstrap'].get(sg, np.array([]))
                b = res['bootstrap'].get(best_sg, np.array([]))
                p_value, _ = _significance_test(a, b)
                significance = ('***' if p_value < 0.001 else
                                '**' if p_value < 0.01 else
                                '*' if p_value < 0.05 else '')

            rows.append({'factor': variable,
                         'subgroup': sg,
                         'metric_value': val,
                         'n_samples': res['counts'][sg],
                         'p_value': p_value,
                         'significance': significance,
                         'is_best': sg == best_sg})

    df_out = pl.DataFrame(rows)
    out = Path(cfg.config.save_path)
    csv_path = out / 'metrics.csv'
    df_out.to_csv(csv_path, index=False)
    logger.info(f"  Metrics CSV saved: {csv_path}")
    return str(csv_path)

def save_experiment_results(bias_results, cfg, global_perfomance, df=None):
    paths = {}

    plot_radar_main(bias_results, cfg)

    paths['radar_main'], paths['bias_index'], paths['mean_abs_gap'], paths['mean_rel_gap'] = plot_radar_main(bias_results, global_perfomance, cfg)
    paths['radar_comparisons'] = plot_all_comparison_radars(bias_results, cfg)
    paths['metrics_csv'] = save_metrics_csv(bias_results, cfg, df=df)

    n_files = len(paths['radar_comparisons']) + 3
    logger.info(f"\n{'='*70}")
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Output directory : {cfg.config.save_path}")
    logger.info(f"Files saved      : {n_files}")
    logger.info(f"Mean |worst - best| (all variables) : {paths['mean_abs_gap']:.3f}  ({paths['mean_rel_gap']:.1f}%)")
    logger.info(f"{'='*70}\n")
    return paths

def run_analysis(cfg, pred_df, population_parquet):
    out = Path(cfg.config.save_path)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=cfg.config.save_path + '/analysis.log', filemode='w', level=logging.INFO)

    logger.info("\nSTEP 1: Preparing variables for analysis...")
    full_df = merge_dfs(pred_df, population_parquet)
    df = prepare_columns(full_df, cfg)
    
    logger.info("\nSTEP 2: Evaluating overall performance...")
    global_performance = compute_global_performance(df, metric=cfg.metric)
    
    logger.info(f"\nSTEP 3: Computing bias metrics for {cfg.metric}...")
    bias_results = compute_all_bias_metrics_classification(df, cfg)
    
    logger.info("\nSTEP 4: Generating visualizations...")
    file_paths = save_experiment_results(bias_results, cfg, global_performance, df=df)
    


