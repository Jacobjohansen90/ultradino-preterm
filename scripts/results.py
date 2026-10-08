import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, roc_curve
import argparse
from tqdm import tqdm

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

output_path = f"/users/data/UCPH/DeepFetal/projects/preterm/misc/{cutoff}.xlsx"

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
              "C-section": "c-section_during_birth"}


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

# ---------------------------------------------------------------------
# 5. Print results
# ---------------------------------------------------------------------

from openpyxl import load_workbook
from copy import copy


# ---------------------------------------------------------------------
# 5. Create Excel output
# ---------------------------------------------------------------------

template_path = "/path/to/Results.xlsx"
output_path = f"/path/to/Results_GA{cutoff}.xlsx"

wb_template = load_workbook(template_path)
ws_template = wb_template["Metrics"]

# Create a new workbook using the existing Metrics sheet as template
wb = load_workbook(template_path)
ws = wb["Metrics"]

# Clear existing Metrics sheet
for row in ws.iter_rows():
    for cell in row:
        cell.value = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def format_metric(r, metric, ci_lower, ci_upper):
    return (
        f"{r[metric]:.3f} "
        f"({r[ci_lower]:.3f}-{r[ci_upper]:.3f})"
    )


def write_result_row(ws, row, start_col, subgroup_name, result):
    ws.cell(row=row, column=start_col, value=subgroup_name)

    ws.cell(
        row=row,
        column=start_col + 1,
        value=format_metric(
            result,
            "auc",
            "auc_ci_lower",
            "auc_ci_upper",
        ),
    )

    ws.cell(
        row=row,
        column=start_col + 2,
        value=format_metric(
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


# ---------------------------------------------------------------------
# Copy formatting from the first GA block in the template
# ---------------------------------------------------------------------

# The template has the desired formatting already.
# We use the first GA block (rows 1-17) as the formatting template.

for row in range(1, 18):
    for col in range(1, 11):
        source = ws_template.cell(row=row, column=col)
        target = ws.cell(row=row, column=col)

        if source.has_style:
            target._style = copy(source._style)

        if source.number_format:
            target.number_format = source.number_format

        if source.alignment:
            target.alignment = copy(source.alignment)

        if source.border:
            target.border = copy(source.border)

        if source.fill:
            target.fill = copy(source.fill)

        if source.font:
            target.font = copy(source.font)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

ws["A1"] = f"GA {cutoff}"

ws["A2"] = "Model"

ws["A3"] = "All patients"
ws["F3"] = "Non-treatment"

ws["B4"] = "AUC"
ws["C4"] = "Sens@Spec"
ws["D4"] = "N-preterm"
ws["E4"] = "N-total"

ws["G4"] = "AUC"
ws["H4"] = "Sens@Spec"
ws["I4"] = "N-preterm"
ws["J4"] = "N-total"


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

subgroup_rows = {
    "All": 5,
    "PPROM": 6,
    "C-Section": 7,
}

for subgroup_name, row in subgroup_rows.items():

    write_result_row(
        ws,
        row,
        1,
        subgroup_name,
        results["All"][subgroup_name]["Model"],
    )

    write_result_row(
        ws,
        row,
        6,
        subgroup_name,
        results["Non-treated"][subgroup_name]["Model"],
    )


# ---------------------------------------------------------------------
# Cervical Length
# ---------------------------------------------------------------------

ws["A8"] = "Cervical Length"

ws["A9"] = "All patients (CL Available)"
ws["F9"] = "Non-treatment (CL Available)"

ws["B10"] = "AUC"
ws["C10"] = "Sens@Spec"
ws["D10"] = "N-preterm"
ws["E10"] = "N-total"

ws["G10"] = "AUC"
ws["H10"] = "Sens@Spec"
ws["I10"] = "N-preterm"
ws["J10"] = "N-total"

cl_rows = {
    "All": 11,
    "PPROM": 12,
    "C-Section": 13,
}

for subgroup_name, row in cl_rows.items():

    write_result_row(
        ws,
        row,
        1,
        subgroup_name,
        results["All"][subgroup_name]["CL"],
    )

    write_result_row(
        ws,
        row,
        6,
        subgroup_name,
        results["Non-treated"][subgroup_name]["CL"],
    )


# ---------------------------------------------------------------------
# Model — CL patients
# ---------------------------------------------------------------------

ws["A14"] = "Model (CL patients)"

ws["A15"] = "All patients (CL Available)"
ws["F15"] = "Non-treatment (CL Available)"

ws["B16"] = "AUC"
ws["C16"] = "Sens@Spec"
ws["D16"] = "N-preterm"
ws["E16"] = "N-total"

ws["G16"] = "AUC"
ws["H16"] = "Sens@Spec"
ws["I16"] = "N-preterm"
ws["J16"] = "N-total"

model_cl_rows = {
    "All": 17,
    "PPROM": 18,
    "C-Section": 19,
}

# Need formatting for rows 18-19 as well
for row in [18, 19]:
    for col in range(1, 11):
        source = ws_template.cell(row=6 + (row - 18), column=col)
        target = ws.cell(row=row, column=col)

        if source.has_style:
            target._style = copy(source._style)


for subgroup_name, row in model_cl_rows.items():

    write_result_row(
        ws,
        row,
        1,
        subgroup_name,
        results["All"][subgroup_name]["Model (CL available)"],
    )

    write_result_row(
        ws,
        row,
        6,
        subgroup_name,
        results["Non-treated"][subgroup_name]["Model (CL available)"],
    )


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

wb.save(output_path)

print(f"Saved: {output_path}")