#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:54:23 2026

@author: jacob
"""

import polars as pl
from omegaconf import OmegaConf, ListConfig
import logging
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter


def bias_analysis(model_path, bias_cfg_path):
    save_path = Path(model_path + 'bias_analysis/')
    save_path.mkdir()
    
    logging.basicConfig(filename=save_path / 'bias.log', filemode='w', level=logging.INFO)
    logger = logging.getLogger('')

    cfg = OmegaConf.load(bias_cfg_path)    
    bias_df = pl.read_csv(cfg.paths.bias_variables, ignore_errors=True)
    
    wb = Workbook()
    wb.remove(wb.active)
    
    for path in sorted((Path(model_path) / "predictions").iterdir()):
        if path.name.endswith(".parquet"):
            GA = path.stem.split("_")[-1]
            predictions = pl.read_parquet(path)
            
            df = predictions.join(bias_df, how='left', left_on='CPR_CHILD', right_on='b_cpr')
    
            if df.height != predictions.height:
                logger.warning(f"WARNING: {predictions.height - df.height} children missing from bias analysis")
            
            results = {}
            orders = {}
            for var in tqdm(cfg.variables[:2], desc=f"Processing Variables, GA {GA}"):
                if isinstance(var.variable, (list, ListConfig)):
                    df = df.with_columns(pl.coalesce([pl.col(var.variable[0]), pl.col(var.variable[1])]).alias(var.variable[0]))
                    var.variable = var.variable[0]
        
                results[var.name] = {}
        
                for pop, pop_df in [("All Births", df), ("Non-treated", df.filter(~pl.col("treatment")))]:
        
                    var_df = pop_df.select(['CPR_CHILD', 'preds', 'preds_binary', 'label', 'threshold', var.variable])
        
                    var_df = handle_nulls(var_df, var)
                    var_df, order = categorize_variable(var_df, var)

                    results[var.name][pop] = {}
                    results[var.name][pop]["Overall"] = calculate_metrics(var_df)

                    for category_df in var_df.partition_by(var.variable, maintain_order=True):
                        category = category_df[var.variable][0]
                        results[var.name][pop][category] = calculate_metrics(category_df, var_df)
                    orders[var.name] = ['Overall'] + order
            wb = to_excel(wb, results, orders, GA)
    wb.save(Path(save_path) / "bias_analysis.xlsx")


def calculate_metrics(df, pop_df=None, n_permutations=10000):
    labels = df["label"].to_numpy().astype(bool)
    preds = df["preds"].to_numpy()
    preds_binary = df["preds_binary"].to_numpy().astype(bool)

    n = len(labels)
    n_preterm = labels.sum()
    n_term = n - n_preterm

    tp = np.sum(preds_binary & labels)
    tn = np.sum(~preds_binary & ~labels)

    N = f"{n}"
    N_preterm = f"{n_preterm}" 
    sens = f"{round(100*tp / n_preterm, 1) if n_preterm else np.nan}"
    spec = f"{round(100*tn / n_term, 1) if n_term else np.nan}"
    AUC = f"{round(roc_auc_score(labels, preds), 3) if n_preterm and n_term else np.nan}"
    sens_p = '-'
    spec_p = '-'
    missed = '-'
    extra = '-'
    
    if pop_df is not None:
        result = calculate_difference(df, pop_df, n_permutations=n_permutations)
        N += f" ({round(100*n/len(pop_df), 1)}%)"
        N_preterm += f" ({round(100*n_preterm/pop_df['label'].sum(), 1)}%)"
        sens += f" ({round(100*result['sens']['difference'], 1)}%)"
        sens_p = f"{round(result['sens']['p_value'], 3)}"
        spec += f" ({round(100*result['spec']['difference'], 1)}%)"
        spec_p = f"{round(result['spec']['p_value'], 3)}"
        AUC += f" ({round(result['AUC']['difference'], 3)})"
        
        # Missed preterms per 100,000 births relative to overall sensitivity
        overall_labels = pop_df["label"].to_numpy().astype(bool)
        overall_preds_binary = pop_df["preds_binary"].to_numpy().astype(bool)
        
        overall_n = len(overall_labels)
        overall_n_preterm = overall_labels.sum()
        overall_n_term = overall_n - overall_n_preterm
        
        overall_tp = np.sum(overall_preds_binary & overall_labels)
        overall_tn = np.sum(~overall_preds_binary & ~overall_labels)
        
        overall_sens = overall_tp / overall_n_preterm if overall_n_preterm else np.nan
        overall_spec = overall_tn / overall_n_term if overall_n_term else np.nan
        
        category_prevalence = n_preterm / n if n else np.nan
        overall_prevalence = overall_n_preterm / overall_n if overall_n else np.nan
        
        if n and np.isfinite(overall_sens) and np.isfinite(overall_prevalence):
            missed = round(100000 * category_prevalence * (overall_sens - tp / n_preterm)) if n_preterm else "-"
        
        if n and np.isfinite(overall_spec) and n_term:
            extra = round(100000 * (1 - category_prevalence) * ((1 - tn / n_term) - (1 - overall_spec)))
    results = {"N": f"{N}",
               "Preterm": f"{N_preterm}",
               "AUC": AUC,
               "Sensitivity": sens,
               "Missed / 100k": missed,
               "P (sens)": sens_p,
               "Specificity": spec,
               "Extra FP / 100k": extra,
               "P (Spec)": spec_p}
    
    return results


def categorize_variable(df, var):
    if var.type == "interval":
        categories = list(var.categories)

        order = ([f"<{categories[0]}"]
                 + [f"{lower}-{upper}" for lower, upper in zip(categories[:-1], categories[1:])]
                 + [f"{categories[-1]}+"])

        expr = pl.when(pl.col(var.variable) < categories[0])
        expr = expr.then(pl.lit(order[0]))

        for lower, upper, label in zip(categories[:-1], categories[1:], order[1:]):
            expr = expr.when(pl.col(var.variable) < upper).then(pl.lit(label))

        expr = expr.otherwise(pl.lit(order[-1]))

        return df.with_columns(expr.alias(var.variable)), order
    
    elif var.type == "binary":
        categories = list(var.categories)
    
        if len(categories) != 2:
            raise ValueError(f"Binary variable '{var.variable}' must have exactly 2 categories")
    
        order = categories
    
        expr = (pl.when(pl.col(var.variable) == True)
                .then(pl.lit(categories[0]))
                .when(pl.col(var.variable) == False)
                .then(pl.lit(categories[1]))
                .otherwise(None))
    
        return df.with_columns(expr.alias(var.variable)), order
    
    elif var.type == 'centiles':
        categories = list(var.categories)

        order = ([f"0th-{categories[0]}th"] 
                 + [f"{lower}th-{upper}th" for lower, upper in zip(categories[:-1], categories[1:])]
                 + [f"{categories[-1]}th-100th"])
       
        percentile_values = df.select([pl.col(var.variable).quantile(p / 100).alias(f"p{p}") for p in categories]).row(0) 
        expr = pl.when(pl.col(var.variable) < percentile_values[0]).then(pl.lit(order[0]))
        
        for value, label in zip(percentile_values[1:], order[1:]):
            expr = expr.when(pl.col(var.variable) < value).then(pl.lit(label))

        expr = expr.otherwise(pl.lit(order[-1]))

        return df.with_columns(expr.alias(var.variable)), order
    
    else:
        raise Exception(f"Type {var.type} not implemented")
        
        
def handle_nulls(df, var):
    if var.nulls == "remove":
        return df.filter(pl.col(var.variable).is_not_null())

    elif var.nulls == "keep":
        return df

    else:
        raise ValueError(f"Unknown null handling '{var.nulls}'\n"
                         "for variable '{var.name}'")
        
        
def calculate_difference(category_df, pop_df, n_permutations=10000):
    rng = np.random.default_rng()
    results = {}

    in_category = pop_df["CPR_CHILD"].is_in(category_df["CPR_CHILD"].implode()).to_numpy()

    # Subgroup AUC using continuous predictions
    subgroup_labels = category_df["label"].to_numpy().astype(bool)
    subgroup_preds = category_df["preds"].to_numpy()
    
    overall_labels = pop_df["label"].to_numpy().astype(bool)
    overall_preds = pop_df["preds"].to_numpy()
    
    subgroup_auc = (roc_auc_score(subgroup_labels, subgroup_preds)
                    if len(np.unique(subgroup_labels)) == 2 else np.nan)
    
    overall_auc = (roc_auc_score(overall_labels, overall_preds)
                   if len(np.unique(overall_labels)) == 2 else np.nan)
    
    results["AUC"] = {"difference": subgroup_auc - overall_auc
                      if np.isfinite(subgroup_auc) and np.isfinite(overall_auc)
                      else np.nan}

    for metric, label_value in [("sens", True), ("spec", False)]:
        labels = pop_df["label"].to_numpy().astype(bool)
        preds = pop_df["preds_binary"].to_numpy().astype(bool)
    
        mask = labels == label_value
    
        subgroup = preds[mask & in_category]
        overall = preds[mask]
    
        if len(subgroup) == 0 or len(overall) == 0:
            results[metric] = {"difference": np.nan,
                               "p_value": np.nan}
            continue
        
        overall_mean = overall.mean()
    
        observed_diff = subgroup.mean() - overall_mean
    
        n_subgroup = len(subgroup)
        permuted_diffs = np.empty(n_permutations)
    
        for i in range(n_permutations):
            shuffled = rng.permutation(overall)
            permuted_diffs[i] = (shuffled[:n_subgroup].mean() - overall_mean)
    
        p_value = (1 + np.sum(np.abs(permuted_diffs) >= abs(observed_diff))) / (n_permutations + 1)

        results[metric] = {"difference": observed_diff,
                           "p_value": p_value}
    
    return results

   
def to_excel(wb, results, orders, GA, significance_level=0.05):
    def format_percent(value):
        return "-" if np.isnan(value) else f"{value:.1%}"

    def format_percent_diff(value):
        return "-" if np.isnan(value) else f"({value:+.1%})"

    def format_value(value, round_value=False):
        if value is None or not np.isfinite(value):
            return "-"
        return round(value) if round_value else value
    
    ws = wb.create_sheet(title=f"GA {GA}")

    section_font = Font(size=14, bold=True, color="FFFFFF")
    header_font = Font(bold=True)

    section_fill_gray = PatternFill("solid", fgColor="595959")
    header_fill_gray = PatternFill("solid", fgColor="B7B7B7")
    results_fill_gray = PatternFill("solid", fgColor="EDEDED")

    section_fill_blue = PatternFill("solid", fgColor="2F5597")
    header_fill_blue = PatternFill("solid", fgColor="8FAADC")
    results_fill_blue = PatternFill("solid", fgColor="EAF2F8")

    red_font = Font(color="9C0006", bold=True)
    green_font = Font(color="006100", bold=True)
    
    thin_border_color_gray = "D9D9D9"
    thick_border_color_gray = "595959"
    thin_border_gray = Side(style="thin", color=thin_border_color_gray)
    thick_border_gray = Side(style="medium", color=thick_border_color_gray)

    thin_border_color_blue = "B4C7E7"
    thick_border_color_blue = "2F5597"
    thin_border_blue = Side(style="thin", color=thin_border_color_blue)
    thick_border_blue = Side(style="medium", color=thick_border_color_blue)

    
    row = 1
    
    for variable_index, (variable, population_data) in enumerate(results.items()):
        if variable_index == 0:
            headers = ['Category'] + list(results[variable]["All Births"]['Overall'].keys())
            columns = len(headers)
        if variable_index % 2 == 0:
            section_fill = section_fill_gray
            header_fill = header_fill_gray
            results_fill = results_fill_gray
            thin_side = thin_border_gray
            thick_side = thick_border_gray
        else:
            section_fill = section_fill_blue
            header_fill = header_fill_blue
            results_fill = results_fill_blue
            thin_side = thin_border_blue
            thick_side = thick_border_blue
                
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=columns*2)

        cell = ws.cell(row=row, column=1, value=variable)
        cell.font = section_font
        cell.fill = section_fill
        cell.alignment = Alignment(horizontal="center")

        row += 1
        
        section_start_row = row
        
        for i, population in enumerate(results[variable].keys()):
            start_col = i * columns + 1
            end_col = start_col + columns - 1

            ws.merge_cells(start_row=row,
                           start_column=start_col,
                           end_row=row,
                           end_column=end_col)

            cell = ws.cell(row=row, column=start_col, value=population)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")
        
        row += 1

        for i in range(2):
            start_col = i * columns + 1
        
            for j, header in enumerate(headers):
                cell = ws.cell(row=row, column=start_col + j, value=header)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center")
        
        row += 1
        significance_threshold = significance_level / (len(results[variable]["All Births"]) - 1)

        for entry in orders[variable]:        
            for i, population in enumerate(results[variable].keys()):
                start_col = i * columns + 1
                data = results[variable][population][entry]
        
                # Write the entry name
                cell = ws.cell(row=row, column=start_col, value=entry)
                cell.fill = results_fill
        
                # Write the remaining values
                for j, header in enumerate(headers[1:], start=1):
                    cell = ws.cell(row=row, column=start_col + j, value=data[header])
                    cell.fill = results_fill
        
                # Color significant sensitivity/specificity differences
                if entry != "Overall":
                    overall = results[variable][population]["Overall"]
        
                    for p_key, metric in [("P (sens)", "Sensitivity"), ("P (Spec)", "Specificity")]:
                        p_value = data[p_key]
        
                        if p_value == "-" or float(p_value) >= significance_threshold:
                            continue
        
                        category_value = data[metric]
                        overall_value = overall[metric]
                        
                        if category_value == "-" or overall_value == "-":
                            continue
                        
                        category_value = float(str(category_value).split()[0])
                        overall_value = float(str(overall_value).split()[0])
        
                        if category_value < overall_value:
                            font = red_font
                        elif category_value > overall_value:
                            font = green_font
                        else:
                            continue
        
                        p_col = start_col + headers.index(p_key)
                        ws.cell(row=row, column=p_col).font = font
        
            row += 1
            
        section_end_row = row - 1
        
        for i in range(2):
            start_col = i * columns + 1
            end_col = start_col + columns - 1
        
            for r in range(section_start_row, section_end_row + 1):
                for c in range(start_col, end_col + 1):
                    cell = ws.cell(row=r, column=c)
        
                    cell.border = Border(left=thick_side if c == start_col else thin_side,
                                         right=thick_side if c == end_col else thin_side,
                                         top=thick_side if r == section_start_row else thin_side,
                                         bottom=thick_side if r == section_end_row else thin_side)
        
    for column in ws.columns:
        max_length = max(len(str(cell.value)) if cell.value is not None else 0
        for cell in column)

        column_letter = get_column_letter(column[0].column)
        ws.column_dimensions[column_letter].width = max_length + 2

    return wb
    