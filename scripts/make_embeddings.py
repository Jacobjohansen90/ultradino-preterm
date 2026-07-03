#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export vision embeddings and per-image predictions for a finished training run.

Picks the best checkpoint from test_metrics.csv in the run folder, then runs
inference on the configured train and test population files.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataloader import (
    PreTermDataset,
    collate_fn,
    load_dataframe,
    prepare_dataframe,
    resolve_naming,
)
from utils.model_utils import model_from_conf

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")


def normalize_run_path(path):
    return path if path.endswith(os.sep) else path + os.sep


def best_checkpoint(run_path):
    """Return weights path and metrics row for the best test checkpoint."""
    metrics_path = run_path + "test_metrics.csv"
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Missing {metrics_path}. Run train.py through test evaluation first."
        )

    result_df = pl.read_csv(metrics_path)
    row = (
        result_df.with_columns(
            max_val=pl.max_horizontal("SensAtSpec_avg", "SensAtSpec_max")
        )
        .sort("max_val", descending=True)
        .head(1)
    )

    weights = row["weights"].item()
    if not str(weights).endswith(".pth"):
        weights = run_path + "weights/" + weights
    elif not os.path.exists(weights):
        weights = run_path + "weights/" + os.path.basename(weights)

    if not os.path.exists(weights):
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    return weights, row


def apply_dataset_filters(df, cfg):
    cols = resolve_naming(cfg)
    ga_weeks_col = cols["GA_WEEKS"]

    df = df.with_columns(pl.lit(False).alias("relabel"))

    for col, cond in cfg.dataset.items():
        if cond == "ignore":
            continue
        if cond == "label":
            df = df.with_columns((pl.col("relabel") | pl.col(col)).alias("relabel"))
        elif cond == "remove":
            df = df.filter(~pl.col(col))
        elif cond == "remove_on_GA":
            df = df.filter(
                ~(pl.col(col) & (pl.col(ga_weeks_col) < cfg.data.ga_cutoff_weeks))
            )

    return df


def load_population(data_path, cfg, split):
    df = prepare_dataframe(load_dataframe(data_path, cfg, split=split), cfg)
    return apply_dataset_filters(df, cfg)


def output_name(split, data_path):
    parts = data_path.rstrip("/").split("/")
    suffix = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return f"{split}_{suffix}.parquet"


def run_embeddings(run_path, output_dir=None, batch_size=None, workers=None):
    run_path = normalize_run_path(run_path)
    output_dir = normalize_run_path(output_dir or run_path + "embeddings")

    cfg = OmegaConf.load(run_path + "conf.yaml")
    weights, best_row = best_checkpoint(run_path)

    print(
        f"Best checkpoint: {weights} "
        f"(SensAtSpec_avg={best_row['SensAtSpec_avg'].item()}, "
        f"SensAtSpec_max={best_row['SensAtSpec_max'].item()})"
    )

    os.makedirs(output_dir, exist_ok=True)

    model = model_from_conf(cfg)
    model.load_state_dict(torch.load(weights, weights_only=True, map_location=cfg.device.type))
    model.eval()

    batch_size = batch_size or cfg.data.batch_size
    workers = workers if workers is not None else cfg.data.workers
    image_path_col = resolve_naming(cfg)["IMAGE_PATH"]

    for split, data_path in [("train", cfg.data.path), ("test", cfg.data.test_path)]:
        df = load_population(data_path, cfg, split=split)
        dataset = PreTermDataset(df, cfg, train=False, ID=image_path_col)

        loader = DataLoader(
            dataset,
            batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=workers,
            collate_fn=collate_fn,
        )

        dfs = []
        embeddings = {}

        for data in tqdm(loader, desc=split):
            with torch.no_grad():
                outputs = model(
                    data["imgs"].to(cfg.device.type),
                    data["img_data"].to(cfg.device.type),
                    data["ehr_data"].to(cfg.device.type),
                    patient_ids=data["IDs"],
                )

                dfs.append(
                    pl.DataFrame(
                        {
                            "img": data["IDs"],
                            "preterm_pred": outputs["preterm"].detach().flatten().cpu().numpy(),
                            "preterm_label": data["labels"]["preterm"].flatten().cpu().numpy(),
                        }
                    )
                )

                embeddings.update(
                    {
                        img_id: emb
                        for img_id, emb in zip(
                            data["IDs"],
                            outputs["vision_features"].detach().to("cpu").tolist(),
                        )
                    }
                )

        pred_df = pl.concat(dfs)
        df_final = df.join(pred_df, left_on=image_path_col, right_on="img", how="left")

        out_name = output_name(split, data_path)
        parquet_path = output_dir + out_name
        json_path = parquet_path.replace(".parquet", ".json")

        df_final.write_parquet(parquet_path)
        with open(json_path, "w") as f:
            json.dump(embeddings, f)

        print(f"Wrote {parquet_path} and {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export embeddings from the best checkpoint in a training run"
    )
    parser.add_argument(
        "run_path",
        help="Path to a Tested run folder containing conf.yaml and test_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for parquet/json outputs (default: <run_path>/embeddings/)",
    )
    parser.add_argument("--batch-size", type=int, help="Override config batch size")
    parser.add_argument("--workers", type=int, help="Override config dataloader workers")
    args = parser.parse_args()

    run_embeddings(args.run_path, args.output_dir, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
