#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:41:00 2026

@author: jacob
"""
#%%Imports
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import polars as pl

from dataloader.dataloader import PreTermDataset, collate_fn, make_train_val_split
from utils.model_utils import model_from_conf, freeze_model
from utils.optim_loader import get_optimizer, get_cosine_schedule_with_warmup
from utils.loss_loader import get_loss
from utils.metric_loader import get_metrics
from utils.documentation import setup, Logger
from utils.test_utils import test_model

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")


def main(config_path):
    #%%Load config and setup logger(s)
    cfg = OmegaConf.load(config_path)

    save_path = setup(cfg)

    logger_avg = Logger(save_path, name='Avg')
    logger_max = Logger(save_path, name='Max')

    #%% Setup dataloaders and models

    train_df, val_df = make_train_val_split(cfg, unique_column='CPR_MOR')
    TrainData = PreTermDataset(train_df, cfg, train=True)
    ValData = PreTermDataset(val_df, cfg, train=False)

    TrainLoader = DataLoader(TrainData,
                             cfg.data.batch_size,
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True,
                             num_workers=cfg.data.workers,
                             collate_fn=collate_fn)

    ValLoader = DataLoader(ValData,
                           cfg.data.batch_size,
                           shuffle=False,
                           pin_memory=False,
                           drop_last=False,
                           num_workers=cfg.data.workers,
                           collate_fn=collate_fn)

    model = model_from_conf(cfg)

    #%% Setup finetuning

    optimizer = get_optimizer(model, cfg)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg, cfg.training.epochs)
    loss_fns = get_loss(cfg)

    metrics_avg = get_metrics(cfg)
    metrics_max = get_metrics(cfg)

    for epoch in range(cfg.training.epochs):
        freeze_model(model, epoch, cfg)

        model.train(True)
        train_loss = 0
        for i, data in enumerate(tqdm(TrainLoader)):
            optimizer.zero_grad()
            outputs = model(data['imgs'].to(cfg.device.type),
                            data['img_data'].to(cfg.device.type),
                            data['ehr_data'].to(cfg.device.type))

            loss = 0
            for task, (_, weight) in cfg.labels.tasks.items():
                loss += loss_fns[task](outputs[task], data['labels'][task].to(cfg.device.type))*weight

            loss.backward()

            train_loss += loss.item() / len(TrainLoader)
            optimizer.step()

        scheduler.step()
        model.eval()

        val_loss = 0
        dfs = []

        with torch.no_grad():
            for data in iter(ValLoader):
                outputs = model(data['imgs'].to(cfg.device.type),
                                data['img_data'].to(cfg.device.type),
                                data['ehr_data'].to(cfg.device.type))

                loss = loss_fns['preterm'](outputs['preterm'].cpu(), data["labels"]["preterm"].cpu())
                val_loss += loss.item() / len(ValLoader)

                dfs.append(pl.DataFrame({'cpr': data['IDs'],
                                         'pred': outputs["preterm"].cpu().squeeze(),
                                         'label': data["labels"]["preterm"].cpu().squeeze()}))

            df = pl.concat(dfs)
            patient_df = (df.group_by("cpr").agg([pl.col("pred").mean().alias("pred_avg"),
                                                  pl.col("pred").max().alias("pred_max"),
                                                  pl.col("label").first().alias("label")]))

            avg_preds = torch.tensor(patient_df['pred_avg'].to_numpy(), dtype=torch.float32)
            max_preds = torch.tensor(patient_df['pred_max'].to_numpy(), dtype=torch.float32)
            labels = torch.tensor(patient_df["label"].to_numpy(), dtype=torch.int)

            for key in metrics_avg.keys():
                metrics_avg[key](avg_preds, labels)
                metrics_max[key](max_preds, labels)

        torch.save(model.state_dict(), save_path + '/weights/' + str(epoch).zfill(3) + '.pth')

        logger_avg.log_metrics(metrics_avg, train_loss, val_loss)
        logger_max.log_metrics(metrics_max, train_loss, val_loss)

        logger_avg.plot_metrics()
        logger_max.plot_metrics()

    test_model(save_path, cfg.data.test_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train preterm prediction model')
    parser.add_argument('config', help='Path to training config YAML file')
    args = parser.parse_args()
    main(args.config)
