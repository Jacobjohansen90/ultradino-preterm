import hydra
from hydra.core.hydra_config import HydraConfig
import lightning.pytorch as pl
import os
from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import logging


from ultradino_sPTB_study.data.pretermbirth_datamodule import PretermDataModule
from ultradino_sPTB_study.model.model import PretermFinetuning, PretermModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="conf", config_name="base", version_base="1.3")
def train(cfg: DictConfig):
    # Access the Hydra output directory - this is where outputs will be stored
    hydra_cfg = HydraConfig.get()
    out_dir = hydra_cfg.runtime.output_dir
    # Notably MLFlow can cause problems if the experiment name contains slashes
    # ("/"), so we replace them here.
    experiment_name = hydra_cfg.job.config_name.replace("/", ".")

    logger.info("Logs and outputs are stored here: %s", out_dir)
    logger.info("*** Configuration: ***")
    logger.info(cfg)

    # Configure loggers to use the Hydra output directory
    # Empty name and version prevent creating additional subdirectories
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(out_dir, "tb_logs"), name="", version=""
    )
    csv_logger = CSVLogger(
        save_dir=os.path.join(out_dir, "csv_logs"), name="", version=""
    )
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        save_dir=os.path.join(out_dir, "mlruns"),
    )

    data = PretermDataModule(csv_dir=cfg.data.csv_dir,
        split_index= cfg.data.split_index,
        label = cfg.data.split_index, #str with column name to take as target
        batch_size = cfg.finetune.batch_size,
        img_size = (224, 224))

    # Configure loss settings
    #assert cfg.loss.type in ["mse", "weighted_re"], f"Unsupported loss type: {cfg.loss.type}"

    model = PretermModel.from_conf(cfg.model)

    finetuning_model = PretermFinetuning(
        model,
        loss_cfg=cfg.loss,
        use_proxy=cfg.model.use_proxy,
        freeze_encoder=cfg.model.freeze_encoder,
        # Optimizer hyperparameters
        optimizer_type=cfg.optimizer.type,
        learning_rate=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        adam_beta1=cfg.optimizer.adam_beta1,
        adam_beta2=cfg.optimizer.adam_beta2,
        adam_eps=cfg.optimizer.adam_eps,
        # Scheduler hyperparameters
        scheduler_type=cfg.scheduler.type,
        patience=cfg.scheduler.patience,
        factor=cfg.scheduler.factor,
        warmup_epochs=cfg.scheduler.warmup_epochs,
        # Learning rate decay parameters
        lr_decay_rate=cfg.scheduler.lr_decay_rate,
        patch_embed_lr_mult=cfg.scheduler.patch_embed_lr_mult,
    )

    # Configure callbacks
    callbacks = []
    if cfg.regularization.early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            monitor=cfg.regularization.early_stopping_metric,
            patience=cfg.regularization.early_stopping_patience,
            mode=cfg.regularization.early_stopping_mode,
            verbose=True,
        )
        callbacks.append(early_stopping)

    if cfg.finetune.checkpoints.every_n_epochs:
        checkpoints_dirpath = cfg.finetune.checkpoints.get(
            "dirpath",
            os.path.join(out_dir, "checkpoints"),
        )
        callbacks.append(
            ModelCheckpoint(
                every_n_epochs=cfg.finetune.checkpoints.every_n_epochs,
                dirpath=checkpoints_dirpath,
                save_last="link",
                # Apparently this is necessary if you want to save each of the
                # checkpoints and not just the last(!) The documentation only
                # reveals this in a very convoluted way. See
                # https://github.com/Lightning-AI/pytorch-lightning/issues/20282
                # and
                # https://stackoverflow.com/q/77032448 for some additional
                # explanation.
                save_top_k=-1,
            )
        )

    profiler = pl.profilers.SimpleProfiler(
        dirpath=os.path.join(out_dir, "profiler"),
        filename="perf_log",
    )

    trainer = pl.Trainer(
        strategy=cfg.device.strategy,
        accelerator=cfg.device.accelerator,
        devices=cfg.device.num_devices,
        max_epochs=cfg.finetune.max_epochs,
        default_root_dir=out_dir,
        logger=[tb_logger, csv_logger, mlf_logger],
        log_every_n_steps=50,
        callbacks=callbacks,
        profiler=profiler,
    )

    trainer.fit(model=finetuning_model, datamodule=data)


if __name__ == "__main__":
    train()
