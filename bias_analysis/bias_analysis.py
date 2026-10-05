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
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment

cfg_path = '/home/jacob/Desktop/NAS/Work/bias_analysis.yaml'


def bias_analysis(cfg_path, model_path):
    cfg = OmegaConf.load(cfg_path)

    logging.basicConfig(filename=model_path + 'bias_analysis/log.log', filemode='w', level=logging.INFO)
    logger = logging.getLogger('')

    
    bias_df = pl.read_csv(cfg.paths.bias_variables, ignore_errors=True)
    predictions = pl.read_parquet('/home/jacob/Desktop/NAS/Work/predictions_34.parquet') #Temporary dummy loader
    
    df = predictions.join(bias_df, how='left', left_on='cpr', right_on='b_cpr')
    
    if df.height != predictions.height:
        logger.warning(f"WARNING: {predictions.height - df.height} children missing from bias analysis")
    
    results = {}
    
    for var in cfg.variables:
        if isinstance(var.variable, (list, ListConfig)):
            df = df.with_columns(pl.coalesce([pl.col(var.variable[0]), pl.col(var.variable[1])]).alias(var.variable[0]))
            var.variable = var.variable[0]

        results[var.name] = {}

        for population, population_df in [("All population", df), ("Non-treated", df.filter(~pl.col("treatment")))]:

            var_df = population_df.select(['cpr', 'preds', 'label', var.variable])

            var_df = handle_nulls(var_df, var)
            var_df, order = categorize_variable(var_df, var)

            sensitivity, specificity, threshold = sens_at_spec(var_df['preds'], var_df['label'])

            results[var.name][population] = {"n": len(var_df),
                                             "sensitivity": sensitivity,
                                             "specificity": specificity,
                                             "categories": []}

            for category in order:
                var_df_category = var_df.filter(pl.col(var.variable) == category)
                summary = permutation_test(var_df, var_df_category, threshold)
                summary["category"] = category
                results[var.name][population]["categories"].append(summary)

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

        order = ([f"<{categories[0]}%"] + [f"{lower}-{upper}%" for lower, upper in zip(categories[:-1], categories[1:])]
                 + [f"{categories[-1]}%<"])
       
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
        raise ValueError(f"Unknown null handling '{var.nulls}' "
                         f"for variable '{var.variable}'")
        
def sens_at_spec(predictions, labels, min_specificity=0.85):
    predictions = np.asarray(predictions)
    labels = np.asarray(labels).astype(bool)

    thresholds = np.unique(predictions)

    best_sensitivity = -np.inf
    best_specificity = np.nan
    best_threshold = np.nan

    for threshold in thresholds:
        preds_binary = predictions >= threshold

        tn = np.sum(~preds_binary & ~labels)
        fp = np.sum(preds_binary & ~labels)
        tp = np.sum(preds_binary & labels)
        fn = np.sum(~preds_binary & labels)

        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan

        if specificity >= min_specificity and sensitivity > best_sensitivity:
            best_sensitivity = sensitivity
            best_specificity = specificity
            best_threshold = threshold

    return best_sensitivity, best_specificity, best_threshold

def sensitivity_specificity_at_threshold(predictions, labels, threshold):
    predictions = predictions >= threshold
    labels = labels.astype(bool)

    tp = (predictions & labels).sum()
    fn = (~predictions & labels).sum()
    tn = (~predictions & ~labels).sum()
    fp = (predictions & ~labels).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return sensitivity, specificity


def permutation_test(df, category_df, threshold, n_permutations=2000, seed=None):
    rng = np.random.default_rng(seed)

    overall_preds = df["preds"].to_numpy()
    overall_labels = df["label"].to_numpy()
        
    category_preds = category_df["preds"].to_numpy()
    category_labels = category_df["label"].to_numpy()

    # Observed performance
    category_sens, category_spec = sensitivity_specificity_at_threshold(category_preds,
                                                                        category_labels,
                                                                        threshold)

    overall_sens, overall_spec = sensitivity_specificity_at_threshold(overall_preds,
                                                                      overall_labels,
                                                                      threshold)

    # Missed opportunities / extra FP per 100k pregnancies
    category_prevalence = np.mean(category_labels==1)
    category_fraction = len(category_df) / len(df)

    expected_missed_opportunities = (100000*category_fraction*category_prevalence*(1 - overall_sens))
    
    category_missed_opportunities = (100000*category_fraction*category_prevalence*(1 - category_sens))
    
    missed_opportunities = (category_missed_opportunities - expected_missed_opportunities)
    
    expected_false_positives = (100000*category_fraction*(1 - category_prevalence)*(1 - overall_spec))
    
    category_false_positives = (100000*category_fraction*(1 - category_prevalence)*(1 - category_spec))
    
    extra_false_positives = (category_false_positives - expected_false_positives)


    positive_preds = overall_preds[overall_labels == 1]
    positive_category_preds = category_preds[category_labels == 1]
    
    observed_sens_diff = (category_sens - overall_sens)

    perm_sens = []

    for _ in range(n_permutations):
        indices = rng.choice(len(positive_preds),
                             size=len(positive_category_preds),
                             replace=False)

        random_sens = np.mean(positive_preds[indices] >= threshold)

        perm_sens.append(random_sens - overall_sens)

    sensitivity_p = (np.sum(np.abs(perm_sens) >= abs(observed_sens_diff)) + 1) / (n_permutations + 1)

    negative_preds = overall_preds[overall_labels == 0]
    category_negative_preds = category_preds[category_labels == 0]

    observed_spec_diff = (category_spec - overall_spec)

    perm_spec = []

    for _ in range(n_permutations):
        indices = rng.choice(len(negative_preds),
                             size=len(category_negative_preds),
                             replace=False)

        random_spec = np.mean(negative_preds[indices] < threshold)

        perm_spec.append(random_spec - overall_spec)

    specificity_p = (np.sum(np.abs(perm_spec) >= abs(observed_spec_diff)) + 1) / (n_permutations + 1)

    return {"n": len(category_df),
            "sensitivity": category_sens,
            "sensitivity_difference": observed_sens_diff,
            "sensitivity_p": sensitivity_p,
            "specificity": category_spec,
            "specificity_difference": observed_spec_diff,
            "specificity_p": specificity_p,
            "missed_opportunities": missed_opportunities,
            "extra_false_positives": extra_false_positives}

def save_bias_analysis_excel(results, save_path):
    
    def nan_to_dash(value, round_value=False):
        if np.isnan(value):
            return "-"
        return round(value) if round_value else value
    
    def format_percent(value):
        return "-" if np.isnan(value) else f"{value:.1%}"
    
    def format_percent_diff(value):
        return "-" if np.isnan(value) else f"({value:+.1%})"
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Bias Analysis"

    section_font = Font(size=14, bold=True, color="FFFFFF")
    header_font = Font(bold=True)

    # Gray palette
    section_fill_gray = PatternFill("solid", fgColor="595959")
    header_fill_gray = PatternFill("solid", fgColor="B7B7B7")
    results_fill_gray = PatternFill("solid", fgColor="EDEDED")
    
    # Blue palette
    section_fill_blue = PatternFill("solid", fgColor="2F5597")
    header_fill_blue = PatternFill("solid", fgColor="8FAADC")
    results_fill_blue = PatternFill("solid", fgColor="EAF2F8")

    red_font = Font(color="9C0006", bold=True)
    green_font = Font(color="006100", bold=True)

    thin_border = Border(left=Side(style="thin", color="A6A6A6"),
                         right=Side(style="thin", color="A6A6A6"),
                         top=Side(style="thin", color="A6A6A6"),
                         bottom=Side(style="thin", color="A6A6A6"))

    headers = ["Category",
               "N",
               "Sensitivity",
               "Missed / 100k",
               "Sens. p",
               "Specificity",
               "Extra FP / 100k",
               "Spec. p"]

    populations = [("All population", 1),("Non-treated", 9)]

    row = 1

    variable_tables = []

    for variable_index, (variable, population_data) in enumerate(results.items()):
        n_categories = len(population_data["All population"]["categories"])
        corrected_alpha = 0.05 / n_categories
        print(corrected_alpha)
        variable_start_row = row

        if variable_index % 2 == 0:
            section_fill = section_fill_gray
            header_fill = header_fill_gray
            results_fill = results_fill_gray
            border_color = "595959"
        else:
            section_fill = section_fill_blue
            header_fill = header_fill_blue
            results_fill = results_fill_blue
            border_color = "2F5597"


        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=16)

        cell = ws.cell(row=row, column=1, value=variable)

        cell.font = section_font
        cell.fill = section_fill 
        cell.border = thin_border
        cell.alignment = Alignment(horizontal="center")

        row += 1

        # Population headings
        for population, start_col in populations:

            ws.merge_cells(start_row=row, start_column=start_col, end_row=row, end_column=start_col + 7)

            cell = ws.cell(row=row, column=start_col, value=population)

            cell.font = header_font
            cell.fill = header_fill
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center")

        row += 1

        # Column headers
        for population, start_col in populations:

            for col, header in enumerate(headers, start=start_col):

                cell = ws.cell(row=row, column=col, value=header)

                cell.font = header_font
                cell.fill = header_fill
                cell.border = thin_border
                cell.alignment = Alignment(horizontal="center")

        row += 1

        # Overall rows
        for population, start_col in populations:

            data = population_data[population]

            overall_values = ["Overall",
                              data["n"],
                              f'{data["sensitivity"]:.1%}',
                              "—",
                              "—",
                              f'{data["specificity"]:.1%}',
                              "—",
                              "—"]

            for col, value in enumerate(overall_values, start=start_col):

                cell = ws.cell(row=row, column=col, value=value)

                cell.font = Font()
                cell.fill = results_fill
                cell.border = thin_border
                cell.alignment = Alignment(horizontal="left")

        row += 1

        # Category rows
        categories = population_data["All population"]["categories"]

        for category_index in range(len(categories)):
            for population, start_col in populations:

                summary = population_data[population]["categories"][category_index]

                values = [summary["category"], 
                          summary["n"],
                          (f'{format_percent(summary["sensitivity"])} '
                           f'{format_percent_diff(summary["sensitivity_difference"])}'),
                          nan_to_dash(summary["missed_opportunities"], round_value=True),
                          nan_to_dash(summary["sensitivity_p"]),
                          (f'{format_percent(summary["specificity"])} '
                           f'{format_percent_diff(summary["specificity_difference"])}'),
                          nan_to_dash(summary["extra_false_positives"], round_value=True),
                          nan_to_dash(summary["specificity_p"])]

                for col, value in enumerate(values, start=start_col):

                    cell = ws.cell(row=row, column=col, value=value)

                    cell.fill = results_fill
                    cell.border = thin_border

                    # P-values
                    if col in [start_col + 4, start_col + 7]:

                        cell.number_format = "0.000"

                        if col == start_col + 4:
                            p_value = summary["sensitivity_p"]
                            difference = summary["sensitivity_difference"]
                        else:
                            p_value = summary["specificity_p"]
                            difference = summary["specificity_difference"]
                    
                        if (not np.isnan(p_value) and p_value < corrected_alpha):
                            if difference < 0:
                                cell.font = red_font
                            elif difference > 0:
                                cell.font = green_font

                    # Numeric alignment
                    if col in [start_col + 1,
                               start_col + 3,
                               start_col + 4,
                               start_col + 6,
                               start_col + 7]:
                        cell.alignment = Alignment(horizontal="left")

            row += 1

        # Last row belonging to this variable
        variable_end_row = row - 1

        variable_tables.append((variable_start_row, variable_end_row, border_color))

    # Column widths
    widths = {"A": 14,
              "B": 10,
              "C": 20,
              "D": 20,
              "E": 12,
              "F": 20,
              "G": 20,
              "H": 12,
              "I": 14,
              "J": 10,
              "K": 20,
              "L": 20,
              "M": 12,
              "N": 20,
              "O": 20,
              "P": 12}

    for column, width in widths.items():
        ws.column_dimensions[column].width = width

    # Borders for all cells in used area
    for row_cells in ws.iter_rows(min_row=1, max_row=ws.max_row,
                                  min_col=1, max_col=16):
        for cell in row_cells:
            cell.border = thin_border

    # Outer borders around each population table
    for start_row, end_row, border_color in variable_tables:

        population_border = Side(style="medium", color=border_color)

        population_divider = Side(style="thick", color=border_color)

        for row_num in range(start_row + 1, end_row + 1):

            # All population: A:H
            ws.cell(row=row_num, column=1).border = Border(left=population_border,
                                                           right=ws.cell(row=row_num, column=1).border.right,
                                                           top=ws.cell(row=row_num, column=1).border.top,
                                                           bottom=ws.cell(row=row_num, column=1).border.bottom)

            ws.cell(row=row_num, column=8).border = Border(left=ws.cell(row=row_num, column=8).border.left,
                                                           right=population_divider,
                                                           top=ws.cell(row=row_num, column=8).border.top,
                                                           bottom=ws.cell(row=row_num, column=8).border.bottom)

            # Non-treated: I:P
            ws.cell(row=row_num, column=9).border = Border(left=population_divider,
                                                           right=ws.cell(row=row_num, column=9).border.right,
                                                           top=ws.cell(row=row_num, column=9).border.top,
                                                           bottom=ws.cell(row=row_num, column=9).border.bottom)

            ws.cell(row=row_num, column=16).border = Border(left=ws.cell(row=row_num, column=16).border.left,
                                                            right=population_border,
                                                            top=ws.cell(row=row_num, column=16).border.top,
                                                            bottom=ws.cell(row=row_num, column=16).border.bottom)

        # Medium bottom border on the last row of this variable
        for col in range(1, 17):
            cell = ws.cell(row=end_row, column=col)

            cell.border = Border(left=cell.border.left, right=cell.border.right, top=cell.border.top, bottom=population_border)

    wb.save(save_path)
