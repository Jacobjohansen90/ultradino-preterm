#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:03:57 2026

@author: jacob
"""

import torchmetrics.classification as tm
import polars as pl
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class Metrics():
    def __init__(self, cfg, save_path):
        self.cutoffs = cfg.tasks.preterm.cutoffs
        self.epoch = 0
        self.metrics = get_metrics(cfg)
        self.dfs = {str(c): [] for c in self.cutoffs}
        self.save_path = save_path
        self.best_score = {str(c): -float("inf") for c in self.cutoffs}
        self.best_predictions = {str(c): None for c in self.cutoffs}
        self.fold = 0

        self.pred_path = Path(self.save_path) / "results" / "predictions" 
        self.metrics_path = Path(self.save_path) / "results" / "metrics"
        self.plot_path = Path(self.save_path) / "results" / "plots"
        
        self.pred_path.mkdir(parents=True, exist_ok=True)
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        self.plot_path.mkdir(parents=True, exist_ok=True)

    def update(self, outputs, data):
        for cutoff in self.cutoffs:
            self.dfs[str(cutoff)].append(pl.DataFrame({'cpr': data['IDs'],
                                                       'preds': outputs['preterm'][str(cutoff)]['preds'].flatten().cpu().numpy(),
                                                       'label': (data['GA_weeks'] < float(cutoff)).flatten().cpu().numpy(),
                                                       'prog': data['progesterone'],
                                                       'remove_on_GA': data['remove_on_GA'].flatten().cpu().numpy()}))
    
    def log_metrics(self, train_loss, test_loss):
        self.epoch += 1
        
        for cutoff in self.cutoffs:
            metrics = {}            

            df = pl.concat(self.dfs[str(cutoff)])
            
            df = df.filter(~(pl.col("remove_on_GA")))

            patient_df = (df.group_by("cpr").agg([pl.col('preds').mean().alias('avg'),
                                                  pl.col('preds').max().alias('max'),
                                                  pl.col('label').first().alias('label'),
                                                  pl.col('prog').first().alias('prog')]))

            labels = torch.tensor(patient_df['label'].to_numpy(), dtype=torch.int32)
            
            for agg in ['avg', 'max']:
                preds = torch.tensor(patient_df[agg].to_numpy(), dtype=torch.float32)
                for metric, name in self.metrics.items():
                    metric.reset()
                    output = metric(preds, labels)
                    
                    if isinstance(output, tuple):
                        metrics[f"{name}_{agg}"] = output[0].item()
                        metrics[f"{name}_cutoff_{agg}"] = output[1].item()
                    else:
                        metrics[f"{name}_{agg}"] = output.item()
                        
            if metrics["SensAtSpec_avg"] >= metrics["SensAtSpec_max"]:
                best_agg = "avg"
                best_score = metrics["SensAtSpec_avg"]
            else:
                best_agg = "max"
                best_score = metrics["SensAtSpec_max"]

            # Keep predictions if this is the best epoch so far
            if best_score > self.best_score[str(cutoff)]:
                self.best_score[str(cutoff)] = best_score

                self.best_predictions[str(cutoff)] = (patient_df.select(["cpr",
                                                                         best_agg,
                                                                         "label",
                                                                         "prog"])
                                                      .rename({best_agg: "preds"}))
                
            row = {'epoch': self.epoch,
                   'train_loss': round(train_loss, 5), 
                   'test_loss': round(test_loss, 5),
                   **dict(sorted(metrics.items()))}

            metrics_df = pl.DataFrame([row])

            path = self.metrics_path / f"metrics_{cutoff}_fold_{self.fold}.csv"
            
            if path.exists():
                existing = pl.read_csv(path)
                metrics_df = pl.concat([existing, metrics_df], how="vertical")

            metrics_df.write_csv(path)

            
            self.plot_metrics(metrics_df, cutoff)
        self.dfs = {str(c): [] for c in self.cutoffs}

    
    def plot_metrics(self, metrics_df, cutoff):
        for agg in ["avg", "max"]:
            
            metric_cols = [col for col in metrics_df.columns if col not in ["epoch", "train_loss", "test_loss"]
                           and "_cutoff_" not in col and col.endswith(f"_{agg}")]
            
            fig, ax = plt.subplots(figsize=(8, 4))
            
            ax.plot(metrics_df["epoch"], metrics_df['train_loss'], label='Train Loss')
            ax.plot(metrics_df["epoch"], metrics_df['test_loss'], label='Test Loss')
            
            for col in metric_cols:
                ax.plot(metrics_df["epoch"], metrics_df[col], label=col)
            
            ax.set_title(f"Fold {self.fold} - GA {cutoff} - {agg.capitalize()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.set_ylim(0, 1.05)
            ax.legend(loc="upper left")
    
            plt.tight_layout()

            fig.savefig(self.plot_path / f"Fold {self.fold} - GA {cutoff} - {agg.capitalize()}.png", dpi=300)
            plt.close(fig)
        
    def reset(self):
        for cutoff in self.cutoffs:
            self.best_predictions[str(cutoff)].write_parquet(self.pred_path / f"predictions_{cutoff}_fold_{self.fold}.parquet")        
 
        self.best_score = {str(c): -float("inf") for c in self.cutoffs}
        self.best_predictions = {str(c): None for c in self.cutoffs}
        self.epoch = 0
        self.dfs = {str(c): [] for c in self.cutoffs}
        self.fold += 1
        
    def summarize_metrics(self):
        for cutoff in self.cutoffs:
            dfs = []
            for fold in range(self.fold):
                path = (Path(self.save_path) / "results" / "metrics" / f"metrics_{cutoff}_fold_{fold}.csv")
                dfs.append(pl.read_csv(path))
    
            df = pl.concat(dfs)
    
            metric_cols = [col for col in df.columns if col != "epoch"]
    
            summary = (df.group_by("epoch").agg([pl.col(col).mean().alias(col)
                                                 for col in metric_cols]).sort("epoch"))
    
            path = (Path(self.save_path) / "results" / f"metrics_{cutoff}_summary.csv")
    
            summary.write_csv(path)
            
    def combine_predictions(self):
        for cutoff in self.cutoffs:
            fold_predictions = []
    
            for fold in range(self.fold):
                path = self.pred_path / f"predictions_{cutoff}_fold_{fold}.parquet"
                
                predictions = (pl.read_parquet(path).with_columns(pl.lit(fold).alias("fold")))
                fold_predictions.append(predictions)
                
            predictions = pl.concat(fold_predictions)

            predictions.write_parquet(self.pred_path / f"predictions_{cutoff}.parquet")
            
        for path in self.pred_path.glob("predictions_*_fold_*.parquet"):
            path.unlink()
            
    def plot_final_metrics(self, results, cutoff):
        metric_names = [name for name in self.metrics]

        x = np.arange(len(metric_names))

        fig, ax = plt.subplots(figsize=(9, 5))
        
        for pop in ['All Births', 'No Progesterone']:
            means = [results[pop][name]["mean"] for name in metric_names]
        
            lower = [results[pop][name]["mean"] - results[pop][name]["lower"] for name in metric_names]
        
            upper = [results[pop][name]["upper"] - results[pop][name]["mean"] for name in metric_names]
        
            ax.errorbar(x,
                        means,
                        yerr=[lower, upper],
                        fmt="o",
                        capsize=5,
                        label=pop)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean across folds")
        ax.set_xlabel("Metric")
        ax.set_title(f"GA {cutoff} weeks")
        ax.legend()
        
        plt.tight_layout()
        
        fig.savefig(self.plot_path / f"GA_{cutoff}_metrics.png", dpi=300)
        
        plt.close(fig)
                    
    def log_final_metrics(self, n_bootstrap=2000):

        self.combine_predictions()
        self.summarize_metrics()

        for cutoff in self.cutoffs:
            report = []
            path = self.pred_path / f"predictions_{cutoff}.parquet"
            df = pl.read_parquet(path)
            
            results = {}

            for pop, pop_df in [("All Births", df), ("No Progesterone", df.filter(~pl.col("prog")))]:
                    
                labels = pop_df["label"].to_numpy()
                preds = pop_df["preds"].to_numpy()
                
                preds_tensor = torch.tensor(preds, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.int32)
                        
                for name, metric in self.metrics.items():
                    metric.reset()
                    output = metric(preds_tensor, labels_tensor)
                    
                    if isinstance(output, tuple):
                        output = output[0]
    
                    point_estimate = output.item()
                    
                    values = []
        
                    for _ in range(n_bootstrap):
                        idx = np.random.randint(0, len(pop_df), len(pop_df))
        
                        boot_preds = torch.tensor(preds[idx], dtype=torch.float32)
        
                        boot_labels = torch.tensor(labels[idx], dtype=torch.int32)
                        
                        metric.reset()
                        output = metric(boot_preds, boot_labels)
        
                        if isinstance(output, tuple):
                            output = output[0]
        
                        values.append(output.item())
        
                    values = np.asarray(values)
        
                    results[pop][name] = {"mean": point_estimate, 
                                          "lower": np.percentile(values, 2.5), 
                                          "upper": np.percentile(values, 97.5)}
        
                n_preterm = int(pop_df["label"].sum())
                n_non_preterm = len(pop_df) - n_preterm
                
                sens_at_spec = self.metrics["SensAtSpec"]
                sens_at_spec.reset()
                
                sens, sens_cutoff = sens_at_spec(preds_tensor, labels_tensor)
            
                report.append(f"--{pop}--\n"
                              f"\n"
                              f"\tPreterm births: {n_preterm}\n"
                              f"\tNon-preterm births: {n_non_preterm}\n"
                              f"\tTotal births: {len(pop_df)}\n"
                              f"\n"
                              f"\tSens@85% specificity: {sens.item():.4f}\n"
                              f"\tSens@85% specificity cutoff: {sens_cutoff.item():.4f}\n"
                              f"\n"
                              f"\tMetrics:\n")
            
                for name, result in results.items():
                    report.append(f"\t\t{name}: {result[pop]['mean']:.4f} (95% CI: {result[pop]['lower']:.4f}–{result[pop]['upper']:.4f})\n")
                report.append('\n\n\n')

            report = "".join(report)
            
            with open(self.metrics_path / f"GA_{cutoff}.txt", "w") as f:
                f.write(report)
            
            self.plot_final_metrics(results, cutoff)
        
def get_metrics(cfg, t=0.5):
    metrics = {'Recall': tm.Recall(task='binary', threshold=t).to(cfg.device.type),
               'Specificity': tm.Specificity(task='binary', threshold=t).to(cfg.device.type),
               'SensAtSpec': tm.SensitivityAtSpecificity(min_specificity=0.85, task='binary').to(cfg.device.type),
               'AUC': tm.AUROC(task='binary').to(cfg.device.type)}
    
    return metrics
    
