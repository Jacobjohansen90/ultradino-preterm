import hydra
from hydra.core.hydra_config import HydraConfig
import lightning.pytorch as pl
import os
from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import logging
import torch

os.environ["CUDA_VISIBLE_DEVICES"]= str(1)

from data.pretermbirth_datamodule import PretermDataModule
from test import test
from model.model import PretermFinetuning, PretermModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="conf", config_name="base_nops", version_base="1.3")
def train(cfg: DictConfig):
    # Access the Hydra output directory - this is where outputs will be stored
    hydra_cfg = HydraConfig.get()
    out_dir = hydra_cfg.runtime.output_dir
    
    # Notably MLFlow can cause problems if the experiment name contains slashes
    # ("/"), so we replace them here.
    experiment_name = "Test_no_ps_best_valloss"#hydra_cfg.job.config_name.replace("/", ".")
       
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
#    mlf_logger = MLFlowLogger(
##        experiment_name=experiment_name,
#        save_dir=os.path.join(out_dir, "mlruns"),
#    )

    data = PretermDataModule(csv_dir=cfg.data.csv_dir,
        split_index= cfg.data.split_index,
        label = cfg.data.label, #str with column name to take as target
        batch_size = cfg.finetune.batch_size,
        num_workers=cfg.finetune.num_workers,
        img_size = (224, 224))

    # Configure loss settings
    #assert cfg.loss.type in ["mse", "weighted_re"], f"Unsupported loss type: {cfg.loss.type}"
    ckpt_path = '/data/proto/Joris/UltraDINO_Bias/src/outputs/base_nops/2025-10-09_10-51-20/checkpoints/epoch=13-step=1834.ckpt'# Replace by the trained model of your choice
    model = PretermModel.from_conf(cfg.model)
    model = PretermFinetuning.load_from_checkpoint(ckpt_path, model=model, map_location='cpu')

    

    # Test
    loader_test = data.setup('test')
    loader_test = data.test_dataloader()
    test(model, loader_test, cfg.data.split_index, out_dir)



if __name__ == "__main__":
    train()
