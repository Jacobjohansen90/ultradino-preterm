import hydra
from hydra.core.hydra_config import HydraConfig
import lightning.pytorch as pl
import os
from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import logging
import pandas as pd
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"]= str(0)

from data.pretermbirth_datamodule import PretermDataModule
#from test_double import test, test_all
#from model.model_double import PretermFinetuning, PretermModel

from test import test, test_all
from model.model import PretermFinetuning, PretermModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#ROOT = '/data/proto/Joris/UltraDINO_Bias/src/outputs/Vit_Small_MaskImg/All_Folds_Spacing_CL2025-11-10_15-49-37/'

#@hydra.main(config_path="conf", config_name="Vit_Small_MaskAlone", version_base="1.3")
@hydra.main(config_path="conf", config_name="Vit_Small_Img_Resampled_B2M_cervical", version_base="1.3")
def train(cfg: DictConfig):
    # Access the Hydra output directory - this is where outputs will be stored
    hydra_cfg = HydraConfig.get()
    out_dir = hydra_cfg.runtime.output_dir
    
    # Notably MLFlow can cause problems if the experiment name contains slashes
    # ("/"), so we replace them here.
    experiment_name = hydra_cfg.job.config_name.replace("/", ".")

    
    
    out_dirs = []
    DF_ULTRA, DF_SONO = [], []
    
    for fold in cfg.data.split_indexes:
        #if fold != 'fold_1':
        #    continue
        # Configure loggers to use the Hydra output directory
        # Empty name and version prevent creating additional subdirectories
        out_dir_current = os.path.join(out_dir, fold)
        
        logger.info("Logs and outputs are stored here: %s", out_dir_current)
        logger.info("*** Configuration: ***")
        logger.info(cfg)
    
        tb_logger = TensorBoardLogger(
            save_dir=os.path.join(out_dir_current, "tb_logs"), name="", version=""
        )
        csv_logger = CSVLogger(
            save_dir=os.path.join(out_dir_current, "csv_logs"), name="", version=""
        )
    #    mlf_logger = MLFlowLogger(
    ##        experiment_name=experiment_name,
    #        save_dir=os.path.join(out_dir, "mlruns"),
    #    )

        data = PretermDataModule(csv_dir=cfg.data.csv_dir,
            split_index= fold,
            label = cfg.data.label, #str with column name to take as target
            input_type = cfg.data.input_type,
            batch_size = cfg.finetune.batch_size,
            num_workers=cfg.finetune.num_workers,
            img_size = (224, 224))

        # Configure loss settings
        #assert cfg.loss.type in ["mse", "weighted_re"], f"Unsupported loss type: {cfg.loss.type}"

        model = PretermModel.from_conf(cfg.model)

        finetuning_model = PretermFinetuning(
            model,
            use_proxy=cfg.model.use_proxy,
            #weight_device=cfg.model.use_proxy,
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
                os.path.join(out_dir_current, "checkpoints"),
            )
            #callbacks.append(
            #    ModelCheckpoint(
            #        every_n_epochs=cfg.finetune.checkpoints.every_n_epochs,
            #        dirpath=checkpoints_dirpath,
            #        save_last="link",
                    # Apparently this is necessary if you want to save each of the
                    # checkpoints and not just the last(!) The documentation only
                    # reveals this in a very convoluted way. See
                    # https://github.com/Lightning-AI/pytorch-lightning/issues/20282
                    # and
                    # https://stackoverflow.com/q/77032448 for some additional
                    # explanation.
            #        save_top_k=-1,
            #    )
            #)
            best_ckp = ModelCheckpoint(monitor='val_AUROC', mode="max", save_top_k=1, dirpath=checkpoints_dirpath, save_last=False)
            #best_ckp = ModelCheckpoint(monitor='val_loss', mode="min", save_top_k=1, dirpath=checkpoints_dirpath, save_last=False)
            callbacks.append(best_ckp)

        profiler = pl.profilers.SimpleProfiler(
            dirpath=os.path.join(out_dir, "profiler"),
            filename="perf_log",
        )

        trainer = pl.Trainer(
            strategy=cfg.device.strategy,
            accelerator=cfg.device.accelerator,
            devices=cfg.device.num_devices,
            max_epochs=cfg.finetune.max_epochs,
            default_root_dir=out_dir_current,
            logger=[tb_logger, csv_logger],
            log_every_n_steps=50,
            callbacks=callbacks,
            profiler=profiler
            
        )
        
        if cfg.model.freeze_encoder_first_blocks != 0:
        
            blocks = finetuning_model.model.encoder_img.blocks[:cfg.model.freeze_encoder_first_blocks]
            
            for b in blocks: 
                for param in b.parameters(): 
                    param.requires_grad = False
        
        trainer.fit(model=finetuning_model, datamodule=data)
        
        # Retrieve best model
        best_model_path = best_ckp.best_model_path #"/data/proto/Joris/UltraDINO_Bias/src/outputs/Vit_Small_MaskAlone/No_Landmarks_Fold_1_Only/fold_1/checkpoints/epoch=9-step=1310.ckpt"#
        model = PretermFinetuning.load_from_checkpoint(best_model_path, model=model, map_location='cpu')
        
        #trainer.validate(model=model, datamodule=data)
        
        # Test
        loader_test = data.setup('test')
        loader_test = data.test_dataloader()
        df_in1, df_in2 = test(model, loader_test, fold, out_dir_current)
        DF_ULTRA.append(df_in1)
        DF_SONO.append(df_in2)
        
        
    DF_ULTRA = pd.concat(DF_ULTRA).reset_index()
    DF_SONO = pd.concat(DF_SONO).reset_index()    
    test_all(DF_ULTRA, DF_SONO, out_dir)



if __name__ == "__main__":
    train()
