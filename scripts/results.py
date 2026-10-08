import argparse

import numpy as np
import polars as pl

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="Name of the model/experiment",
)

args = parser.parse_args()

model = args.model_name


cutoff = model.split('_')[-1]

output_path = f"/users/data/UCPH/DeepFetal/projects/preterm/misc/{model}.xlsx"

def sens_at_85_spec(y_true, y_score, target_spec=0.85):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    specificity = 1 - fpr

    valid = specificity >= target_spec

    if not np.any(valid):
        return np.nan

    return np.max(tpr[valid])


def bootstrap_metrics(
    df,
    score_col,
    lower_is_positive=False,
    n_bootstrap=2000,
    seed=42,
):
    data = df.select(["label", score_col]).drop_nulls()

    y = data["label"].to_numpy()
    scores = data[score_col].to_numpy()

    if lower_is_positive:
        scores = -scores

    # Point estimates
    auc = roc_auc_score(y, scores)
    sens = sens_at_85_spec(y, scores)

    # Bootstrap
    rng = np.random.default_rng(seed)

    auc_boot = []
    sens_boot = []

    n = len(data)

    for _ in range(n_bootstrap):

        idx = rng.integers(0, n, n)

        y_b = y[idx]
        scores_b = scores[idx]

        if len(np.unique(y_b)) < 2:
            continue

        auc_boot.append(
            roc_auc_score(y_b, scores_b)
        )

        sens_boot.append(
            sens_at_85_spec(y_b, scores_b)
        )

    auc_ci = np.percentile(auc_boot, [2.5, 97.5])
    sens_ci = np.percentile(sens_boot, [2.5, 97.5])

    return {
        "n": n,
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "auc": auc,
        "auc_ci_lower": auc_ci[0],
        "auc_ci_upper": auc_ci[1],
        "sens_at_85_spec": sens,
        "sens_ci_lower": sens_ci[0],
        "sens_ci_upper": sens_ci[1],
    }

# ---------------------------------------------------------------------
# 1. Collapse df to one row per child and join with predictions
# ---------------------------------------------------------------------

cutoff = int(model.split('_')[-1])
path = '/users/data/UCPH/DeepFetal/projects/preterm/training_runs/Running/'
preds = pl.read_parquet(path + model + f"/results/predictions/predictions_{cutoff}.parquet")
df = pl.read_parquet('/users/data/UCPH/DeepFetal/projects/preterm/Data/dataset_v6/test2.parquet')

df_combined = (
    preds
    .join(
        df.group_by("CPR_CHILD").agg([
            pl.col("GA").first(),
            pl.col("induced").first(),
            pl.col("c-section").first(),
            pl.col("pprom").first(),
            pl.col("c-section_during_birth").first(),
            pl.col("contractions_with_preterm_birth").first(),
            pl.col("CL").first(),
        ]),
        on="CPR_CHILD",
        how="left",
    )
    .filter(
        (pl.col("GA") // 7 >= cutoff)
        | (~pl.col("induced") & ~pl.col("c-section"))
        | pl.col("pprom")
        | (
            pl.col("c-section")
            & (
                pl.col("c-section_during_birth")
                | pl.col("contractions_with_preterm_birth")
            )
            & ~pl.col("induced")
        )
    )
)


populations = {"All": pl.lit(True),
               "Non-treated": ~pl.col("treatment")}

sub_groups = {"All": None,
              "PPROM": "pprom",
              "C-Section": "c-section_during_birth"}


# ---------------------------------------------------------------------
# 4. Calculate metrics
# ---------------------------------------------------------------------

results = {}

total = len(populations) * len(sub_groups)

with tqdm(total=total, desc="Calculating metrics") as pbar:

    for population_name, population_filter in populations.items():
    
        population_df = df_combined.filter(population_filter)
    
        results[population_name] = {}
    
        for sub_group_name, sub_group_value in sub_groups.items():
    
            if sub_group_value is None:
                subgroup_df = population_df
            else:
                subgroup_df = population_df.filter((pl.col("GA") // 7 >= cutoff)
                                                   | ((pl.col("GA") // 7 < cutoff)
                                                      & pl.col(sub_group_value)))
    
            # -------------------------------------------------------------
            # Model — all patients
            # -------------------------------------------------------------
    
            model_results = bootstrap_metrics(
                subgroup_df,
                score_col="preds",
                lower_is_positive=False,
            )
    
            # -------------------------------------------------------------
            # Patients with a valid CL measurement
            # -------------------------------------------------------------
    
            cl_df = subgroup_df.filter(
                pl.col("CL") != 0
            )
    
            # -------------------------------------------------------------
            # Model — CL available
            # -------------------------------------------------------------
    
            model_cl_results = bootstrap_metrics(
                cl_df,
                score_col="preds",
                lower_is_positive=False,
            )
    
            # -------------------------------------------------------------
            # CL — CL available
            # -------------------------------------------------------------
    
            cl_results = bootstrap_metrics(
                cl_df,
                score_col="CL",
                lower_is_positive=True,
            )
    
            results[population_name][sub_group_name] = {
                "Model": model_results,
                "Model (CL available)": model_cl_results,
                "CL": cl_results,
            }

            pbar.update(1)

wb = Workbook()

ws = wb.active
ws.title = "Metrics"


# =====================================================================
# 9. Excel formatting
# =====================================================================

thin_gray = Side(
    style="thin",
    color="D9D9D9",
)

medium_gray = Side(
    style="medium",
    color="A6A6A6",
)

header_fill = PatternFill(
    fill_type="solid",
    fgColor="D9EAF7",
)

section_fill = PatternFill(
    fill_type="solid",
    fgColor="E7E6E6",
)

subsection_fill = PatternFill(
    fill_type="solid",
    fgColor="F2F2F2",
)

title_font = Font(
    bold=True,
    size=14,
)

section_font = Font(
    bold=True,
    size=11,
)

header_font = Font(
    bold=True,
)

normal_alignment = Alignment(
    horizontal="center",
    vertical="center",
)

left_alignment = Alignment(
    horizontal="left",
    vertical="center",
)

center_alignment = Alignment(
    horizontal="center",
    vertical="center",
)

# =====================================================================
# 10. Excel helper functions
# =====================================================================

def metric_string(
    result,
    value_key,
    lower_key,
    upper_key,
):
    return (
        f"{result[value_key]:.3f} "
        f"({result[lower_key]:.3f}-{result[upper_key]:.3f})"
    )


def write_metric_row(
    ws,
    row,
    start_col,
    subgroup_name,
    result,
):

    ws.cell(
        row=row,
        column=start_col,
        value=subgroup_name,
    )

    ws.cell(
        row=row,
        column=start_col + 1,
        value=metric_string(
            result,
            "auc",
            "auc_ci_lower",
            "auc_ci_upper",
        ),
    )

    ws.cell(
        row=row,
        column=start_col + 2,
        value=metric_string(
            result,
            "sens_at_85_spec",
            "sens_ci_lower",
            "sens_ci_upper",
        ),
    )

    ws.cell(
        row=row,
        column=start_col + 3,
        value=result["n_positive"],
    )

    ws.cell(
        row=row,
        column=start_col + 4,
        value=result["n"],
    )


def format_metric_block(
    ws,
    title_row,
    population_row,
    header_row,
    data_start_row,
    population_left,
    population_right,
):

    # Section title
    ws.merge_cells(
        start_row=title_row,
        start_column=1,
        end_row=title_row,
        end_column=10,
    )

    ws.cell(
        title_row,
        1,
    ).font = section_font

    ws.cell(
        title_row,
        1,
    ).fill = section_fill

    ws.cell(
        title_row,
        1,
    ).alignment = left_alignment

    # Population headers
    ws.merge_cells(
        start_row=population_row,
        start_column=1,
        end_row=population_row,
        end_column=5,
    )

    ws.merge_cells(
        start_row=population_row,
        start_column=6,
        end_row=population_row,
        end_column=10,
    )

    ws.cell(
        population_row,
        1,
        population_left,
    )

    ws.cell(
        population_row,
        6,
        population_right,
    )

    for col in [1, 6]:

        cell = ws.cell(
            population_row,
            col,
        )

        cell.font = header_font
        cell.fill = subsection_fill
        cell.alignment = normal_alignment

    # Column headers
    headers = [
        "Subgroup",
        "AUC",
        "Sens@Spec",
        "N-preterm",
        "N-total",
    ]

    for i, header in enumerate(headers, start=1):

        ws.cell(
            header_row,
            i,
            header,
        )

        ws.cell(
            header_row,
            i,
        ).font = header_font

        ws.cell(
            header_row,
            i,
        ).fill = header_fill

        ws.cell(
            header_row,
            i,
        ).alignment = normal_alignment

        ws.cell(
            header_row,
            i + 5,
            header,
        )

        ws.cell(
            header_row,
            i + 5,
        ).font = header_font

        ws.cell(
            header_row,
            i + 5,
        ).fill = header_fill

        ws.cell(
            header_row,
            i + 5,
        ).alignment = normal_alignment

    # Data rows
    subgroup_names = [
        "All",
        "PPROM",
        "C-Section",
    ]

    for i, subgroup_name in enumerate(
        subgroup_names,
        start=data_start_row,
    ):

        for col in range(1, 11):

            cell = ws.cell(
                i,
                col,
            )

            cell.alignment = normal_alignment

            cell.border = Border(
                bottom=thin_gray,
            )


# =====================================================================
# 11. Title
# =====================================================================

ws["A1"] = f"GA {cutoff}"
ws["A1"].font = Font(bold=True, size=14, color="FFFFFF")
ws["A1"].fill = PatternFill(fill_type="solid", fgColor="595959")
ws["A1"].alignment = center_alignment

ws.merge_cells("A1:J1")


# =====================================================================
# 12. Model section
# =====================================================================

format_metric_block(
    ws,
    title_row=2,
    population_row=3,
    header_row=4,
    data_start_row=5,
    population_left="Model — All patients",
    population_right="Model — Non-treatment",
)

for i, subgroup_name in enumerate(
    ["All", "PPROM", "C-Section"],
    start=5,
):
    write_metric_row(
        ws, i, 1, subgroup_name,
        results["All"][subgroup_name]["Model"],
    )

    write_metric_row(
        ws, i, 6, subgroup_name,
        results["Non-treated"][subgroup_name]["Model"],
    )


# =====================================================================
# 13. Cervical Length section
# =====================================================================

format_metric_block(
    ws,
    title_row=8,
    population_row=9,
    header_row=10,
    data_start_row=11,
    population_left="CL — All patients",
    population_right="CL — Non-treatment",
)

for i, subgroup_name in enumerate(
    ["All", "PPROM", "C-Section"],
    start=11,
):
    write_metric_row(
        ws, i, 1, subgroup_name,
        results["All"][subgroup_name]["CL"],
    )

    write_metric_row(
        ws, i, 6, subgroup_name,
        results["Non-treated"][subgroup_name]["CL"],
    )


# =====================================================================
# 14. Model — CL patients section
# =====================================================================

format_metric_block(
    ws,
    title_row=14,
    population_row=15,
    header_row=16,
    data_start_row=17,
    population_left="Model CL — All patients",
    population_right="Model CL — Non-treatment",
)

for i, subgroup_name in enumerate(
    ["All", "PPROM", "C-Section"],
    start=17,
):
    write_metric_row(
        ws,
        i,
        1,
        subgroup_name,
        results["All"][subgroup_name]["Model (CL available)"],
    )

    write_metric_row(
        ws,
        i,
        6,
        subgroup_name,
        results["Non-treated"][subgroup_name]["Model (CL available)"],
    )


# =====================================================================
# 15. General formatting
# =====================================================================

widths = {
    "A": 18,
    "B": 22,
    "C": 22,
    "D": 14,
    "E": 14,
    "F": 18,
    "G": 22,
    "H": 22,
    "I": 14,
    "J": 14,
}

for column, width in widths.items():
    ws.column_dimensions[column].width = width


# Center all cells
for row in ws.iter_rows(
    min_row=1,
    max_row=ws.max_row,
    min_col=1,
    max_col=10,
):
    for cell in row:
        if cell.value is not None:
            cell.alignment = center_alignment


# Stronger borders around population blocks
for start_row, end_row in [
    (3, 7),
    (9, 13),
    (15, 19),
]:
    for row in range(start_row, end_row + 1):
        for col in [1, 5, 6, 10]:
            ws.cell(row, col).border = Border(
                left=medium_gray if col in [1, 6] else thin_gray,
                right=medium_gray if col in [5, 10] else thin_gray,
                bottom=thin_gray,
            )


# Freeze title
ws.freeze_panes = "A2"


# =====================================================================
# 16. Save
# =====================================================================

wb.save(output_path)

print(f"\nSaved Excel file to: {output_path}")