import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------------
# 1. Collapse df to one row per child and join with predictions
# ---------------------------------------------------------------------

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
            pl.col("CL_read").first(),
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


# ---------------------------------------------------------------------
# 2. Metric functions
# ---------------------------------------------------------------------

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
# 3. Define populations
# ---------------------------------------------------------------------

populations = {
    "All": pl.lit(True),
    "Non-treated": ~pl.col("treatment"),
}

pprom_groups = {
    "All": None,
    "PPROM": True,
    "Non-PPROM": False,
}


# ---------------------------------------------------------------------
# 4. Calculate metrics
# ---------------------------------------------------------------------

results = {}

for population_name, population_filter in populations.items():

    population_df = df_combined.filter(population_filter)

    results[population_name] = {}

    for pprom_name, pprom_value in pprom_groups.items():

        if pprom_value is None:
            subgroup_df = population_df
        else:
            subgroup_df = population_df.filter(
                pl.col("pprom") == pprom_value
            )

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
            pl.col("CL_read") != 0
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
            score_col="CL_read",
            lower_is_positive=True,
        )

        results[population_name][pprom_name] = {
            "Model": model_results,
            "Model (CL available)": model_cl_results,
            "CL": cl_results,
        }


# ---------------------------------------------------------------------
# 5. Print results
# ---------------------------------------------------------------------

for population, population_results in results.items():

    print(f"\n{'=' * 75}")
    print(population)
    print(f"{'=' * 75}")

    for subgroup, metrics in population_results.items():

        print(f"\n{subgroup}")
        print("-" * len(subgroup))

        for method, r in metrics.items():

            print(f"\n  {method}")
            print(f"  N:                  {r['n']:,}")
            print(f"  Positive:            {r['n_positive']:,}")
            print(f"  Negative:            {r['n_negative']:,}")
            print(
                f"  AUC:                 {r['auc']:.3f} "
                f"(95% CI: "
                f"{r['auc_ci_lower']:.3f}–"
                f"{r['auc_ci_upper']:.3f})"
            )
            print(
                f"  Sens. @ 85% spec.:   "
                f"{r['sens_at_85_spec']:.3f} "
                f"(95% CI: "
                f"{r['sens_ci_lower']:.3f}–"
                f"{r['sens_ci_upper']:.3f})"
            )