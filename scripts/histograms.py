#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:40:20 2026

@author: jacob
"""

import argparse

import polars as pl
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="Name of the model/experiment",
)

args = parser.parse_args()

model = args.model_name

bias_df = pl.read_csv('/users/data/UCPH/DeepFetal/projects/preterm/Data/misc/bias_variables.csv')

cutoff = model.split('_')[-1]

output_path = f"/users/data/UCPH/DeepFetal/projects/preterm/misc/{model}/"

cutoff = int(model.split('_')[-1])
path = '/users/data/UCPH/DeepFetal/projects/preterm/training_runs/Running/'


def plot_variable_histograms(
    df: pl.DataFrame,
    variable: str,
    save_path: str,
    bins: int = 30,
):
    # ================================================================
    # 1. Define populations
    # ================================================================

    populations = {
        "all": pl.lit(True),
        "non_treated": ~pl.col("treatment"),
    }

    # ================================================================
    # 2. Define prediction groups
    # ================================================================

    comparisons = {
        "positives": {
            "all_patients": pl.lit(True),
            "true_positives": (
                (pl.col("preds") == 1) & (pl.col("label") == 1)
            ),
            "false_positives": (
                (pl.col("preds") == 1) & (pl.col("label") == 0)
            ),
        },
        "negatives": {
            "all_patients": pl.lit(True),
            "true_negatives": (
                (pl.col("preds") == 0) & (pl.col("label") == 0)
            ),
            "false_negatives": (
                (pl.col("preds") == 0) & (pl.col("label") == 1)
            ),
        },
    }

    # ================================================================
    # 3. Define clinical subgroups
    # ================================================================

    sub_groups = {
        "All": None,
        "PPROM": "pprom",
        "C-Section": "c-section_during_birth",
    }

    # ================================================================
    # 4. Prepare output directory
    # ================================================================

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 5. Prediction group histograms
    # ================================================================

    for population_name, population_filter in populations.items():

        population_df = df.filter(population_filter)

        for comparison_name, groups in comparisons.items():

            all_values = (
                population_df
                .get_column(variable)
                .drop_nulls()
                .to_numpy()
            )

            if len(all_values) == 0:
                continue

            bin_edges = np.histogram_bin_edges(
                all_values,
                bins=bins,
            )

            fig, ax = plt.subplots(figsize=(9, 6))

            for group_name, group_filter in groups.items():

                values = (
                    population_df
                    .filter(group_filter)
                    .get_column(variable)
                    .drop_nulls()
                    .to_numpy()
                )

                ax.hist(
                    values,
                    bins=bin_edges,
                    alpha=0.4,
                    label=(
                        f"{group_name.replace('_', ' ').title()} "
                        f"(n={len(values):,})"
                    ),
                    edgecolor="black",
                    linewidth=0.5,
                )

            ax.set_title(
                f"{variable} — "
                f"{population_name.replace('_', ' ').title()} — "
                f"{comparison_name.title()}"
            )
            ax.set_xlabel(variable)
            ax.set_ylabel("Number of patients")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)

            fig.tight_layout()

            filename = (
                f"{variable}_{population_name}_{comparison_name}.png"
            )

            fig.savefig(
                save_path / filename,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

    # ================================================================
    # 6. Clinical subgroup histograms
    # ================================================================

    for population_name, population_filter in populations.items():

        population_df = df.filter(population_filter)

        for subgroup_name, subgroup_column in sub_groups.items():

            # Skip the "All" subgroup
            if subgroup_column is None:
                continue

            subgroup_df = population_df.filter(
                pl.col(subgroup_column).fill_null(False)
            )

            all_values = (
                population_df
                .get_column(variable)
                .drop_nulls()
                .to_numpy()
            )

            subgroup_values = (
                subgroup_df
                .get_column(variable)
                .drop_nulls()
                .to_numpy()
            )

            if len(all_values) == 0:
                continue

            bin_edges = np.histogram_bin_edges(
                all_values,
                bins=bins,
            )

            fig, ax = plt.subplots(figsize=(9, 6))

            ax.hist(
                all_values,
                bins=bin_edges,
                alpha=0.4,
                edgecolor="black",
                linewidth=0.5,
                label=f"All patients (n={len(all_values):,})",
            )

            ax.hist(
                subgroup_values,
                bins=bin_edges,
                alpha=0.5,
                edgecolor="black",
                linewidth=0.5,
                label=f"{subgroup_name} (n={len(subgroup_values):,})",
            )

            ax.set_title(
                f"{variable} — "
                f"{population_name.replace('_', ' ').title()} — "
                f"{subgroup_name}"
            )
            ax.set_xlabel(variable)
            ax.set_ylabel("Number of patients")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)

            fig.tight_layout()

            # Generate filenames
            subgroup_slug = subgroup_column.replace("-", "_")

            if population_name == "all":
                filename = f"{variable}_{subgroup_slug}.png"
            else:
                filename = (
                    f"{variable}_{population_name}_{subgroup_slug}.png"
                )

            fig.savefig(
                save_path / filename,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)


# =====================================================================
# 1. Load data
# =====================================================================

preds = pl.read_parquet(
    path + model + f"/results/predictions/predictions_{cutoff}.parquet"
)

df = pl.read_parquet(
    "/users/data/UCPH/DeepFetal/projects/preterm/Data/dataset_v6/test2.parquet"
)

# =====================================================================
# 2. Aggregate clinical variables and image-level information
# =====================================================================

clinical_df = df.group_by("CPR_CHILD").agg([
    # Clinical variables
    pl.col("GA").first(),
    pl.col("induced").first(),
    pl.col("c-section").first(),
    pl.col("pprom").first(),
    pl.col("c-section_during_birth").first(),
    pl.col("contractions_with_preterm_birth").first(),
    pl.col("CL").first(),

    # Mean physical pixel spacing across images
    pl.mean_horizontal(
        pl.col("physical_delta_x"),
        pl.col("physical_delta_y"),
    ).mean().alias("physical_delta"),

    # Number of images per child
    pl.len().alias("number_of_imgs"),
])


# =====================================================================
# 3. Merge predictions with clinical variables
# =====================================================================

df_combined = preds.join(
    clinical_df,
    on="CPR_CHILD",
    how="left",
)

# Apply inclusion criteria
df_combined = df_combined.filter(
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


# =====================================================================
# 4. Merge bias variables
# =====================================================================

assert bias_df["b_cpr"].is_unique(), (
    "bias_df contains duplicate b_cpr values; "
    "the join could duplicate prediction rows."
)

df_combined = df_combined.join(
    bias_df,
    left_on="CPR_CHILD",
    right_on="b_cpr",
    how="left",
)


# =====================================================================
# 5. Validate
# =====================================================================

assert df_combined["CPR_CHILD"].is_unique(), (
    "df_combined contains duplicate CPR_CHILD values."
)

print(f"Children after filtering: {df_combined.height:,}")

# =====================================================================
# 6. Model Insight — Variable Histograms
# =====================================================================

variables = [
    "maternal_age",
    "maternal_BMI",
    "physical_delta",
    "number_of_imgs",
]

histogram_path = Path(output_path) / "histograms"

for variable in variables:
    if variable not in df_combined.columns:
        print(f"Skipping {variable}: column not found in df_combined.")
        continue

    print(f"Creating histograms for {variable}...")

    plot_variable_histograms(
        df=df_combined,
        variable=variable,
        save_path=str(histogram_path),
    )

print(f"Histograms saved to: {histogram_path}")






