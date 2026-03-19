import lightning.pytorch as pl
import torch
from torch import nn
from lightning.pytorch.loggers import TensorBoardLogger

from ultradino_finetune.models import (
    load_pretrained_ultradino,
    load_from_scratch,
    get_param_groups_with_decay,
)

from ultradino_preterm_study.utils.lr_scheduler import get_cosine_schedule_with_warmup
import logging


from torchmetrics.functional import auroc, recall, precision, specificity, accuracy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SpacingDecoder(nn.Module):
    
    """ Decoder with FiLM conditioning from pixel spacing"""
    
    def __init__(self, dim_embedding_plane, dim_embedding_ps, outputs):
        super().__init__()

        self.outputs = outputs

        self.gamma_layer = nn.Linear(dim_embedding_ps, dim_embedding_plane)
        self.beta_layer = nn.Linear(dim_embedding_ps, dim_embedding_plane)

        self.fc = nn.Linear(
            dim_embedding_plane,
            len(outputs),
        )        

    def forward(self, x):
        embedding_plane, embedding_spacing = x

        #Perform FiLM conditioning  on the plane embedding with the pixel spacing embedding
        # y = (1+f)*x + b
        # Il faut projeter l'encodage de l'espacement (R8) vers un espace de dimension égale à dim(embedding_plane)
        # Une fois pour le gain et une fois pour le biais
        gamma = self.gamma_layer(embedding_spacing)
        beta = self.beta_layer(embedding_spacing)

        #Pour obtenir une plage stable, on va utiliser tanh et multiplier par un facteur
        #gamma = torch.tanh(gamma) * 3

        # Puis faire un produit Hadamard entre le gain et l'embedding de la plane
        # Et une addition entre le biais et l'embedding de la plane
        x = (gamma) * embedding_plane + beta #dim: batch_size, embedding plane
        #x = torch.cat((embedding_plane, x), -1)
        
        return self.fc(x)

    def output_to_dict(self, y):
        """Use the decoder definition to map array position to output names"""

        # Sanity check
        if not y.size(-1) == len(self.outputs):
            raise RuntimeError(
                f'Number of outputs does not match model definition '
                f'({len(y)}!={len(self.outputs)}'
            )

        #return {'logit': y}
        return {name: y[..., i] for i, name in enumerate(self.outputs)}

class BasicDecoder(nn.Module):

    """ FC Decoder """
    
    def __init__(self, dim_embedding_plane, dim_embedding_ps, outputs):
        super().__init__()

        self.outputs = outputs
        self.fc = nn.Linear(
            dim_embedding_plane,
            len(outputs),
        )

    def forward(self, x):
        embedding_plane, embedding_spacing = x
        x = embedding_plane
        
        return self.fc(x)

    def output_to_dict(self, y):
        """Use the decoder definition to map array position to output names"""

        # Sanity check
        if not y.size(-1) == len(self.outputs):
            raise RuntimeError(
                f'Number of outputs does not match model definition '
                f'({len(y)}!={len(self.outputs)}'
            )

        #return {'logit': y}
        return {name: y[..., i] for i, name in enumerate(self.outputs)}
        
        
class PretermModel(nn.Module):
    
    """ """
    
    IMAGE_DIM = 2

    @classmethod
    def from_conf(cls, cfg):
        """Create growth model from configuration"""

        if cfg.encoder.weights_path:
            logger.info("Loading pretrained encoder from %s", cfg.encoder.weights_path)
            encoder_img = load_pretrained_ultradino(
                cfg.encoder.type,
                cfg.encoder.weights_path,
            )
            encoder_mask = load_pretrained_ultradino(
                cfg.encoder.type,
                cfg.encoder.weights_path,
            )
        else:
            logger.warning(
                "No pretrained weights provided. Initializing encoder randomly."
            )
            # Initialize a new encoder randomly
            encoder_img = load_from_scratch(cfg.encoder.type)
            encoder_mask = load_from_scratch(cfg.encoder.type)

        # input to the decoder is feature vector of the cervix plane + pixel spacing


        if cfg.decoder.type == 'spacing':
            if cfg.landmarks:
                decoder = SpacingDecoder(encoder_img.embed_dim + encoder_mask.embed_dim + 64, cfg.pixel_spacing_embed_dim, cfg.decoder.outputs)
            else:
                decoder = SpacingDecoder(encoder_img.embed_dim + encoder_mask.embed_dim, cfg.pixel_spacing_embed_dim, cfg.decoder.outputs)
            
        elif cfg.decoder.type == 'basic':
            decoder = BasicDecoder(encoder_img.embed_dim + encoder_mask.embed_dim, cfg.pixel_spacing_embed_dim, cfg.decoder.outputs)
#BasicDecoder(num_inputs, cfg.decoder.outputs)
        else:
            raise RuntimeError(f'Unknown decoder type f"{cfg.type}"')

        return cls(
            encoder_img,
            encoder_mask,
            decoder,
            cfg.planes,
            pixel_spacing_embed_dim=cfg.pixel_spacing_embed_dim,
            landmarks=cfg.landmarks,
        )

    def __init__(
        self,
        encoder_img,
        encoder_mask,
        decoder,
        planes,
        pixel_spacing_embed_dim=0,
        landmarks=False
    ):
        super().__init__()

        self.encoder_img = encoder_img
        
        # blocks = self.enc.blocks[:k]
        # for b in blocks: for param in b.parameters: b.requires_grad = False
        
        self.class_head_encoder_img = nn.Linear(encoder_img.embed_dim,1)
        
        self.encoder_mask = encoder_mask
        self.class_head_encoder_mask = nn.Linear(encoder_mask.embed_dim,1)
        self.decoder = decoder
        self.landmarks = landmarks
        self.planes = planes
        
        # Currently only needed for extracting parameters in setup of optimizer
        # and logging learning rate scheduling in finetuning module and in the
        # forward method below.
        self.pixel_spacing_embed_dim = pixel_spacing_embed_dim

        if pixel_spacing_embed_dim:
            self.pixel_spacing_encoder = nn.Sequential(
                nn.Linear(self.IMAGE_DIM, pixel_spacing_embed_dim),
                nn.ReLU(),
            )
        else:
            self.pixel_spacing_encoder = nn.Identity()


        if self.landmarks:
            self.landmarks_encoder = nn.Sequential(
                nn.Linear(45, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            
            self.class_head_landmarks = nn.Linear(
            64,1)

    def has_pixel_spacing_encoder(self):
        return bool(self.pixel_spacing_embed_dim)

    def has_landmarks_encoder(self):
        return self.landmarks
        
    def forward(self, x):
        
        img_embedding = self.encoder_img(x["image"])
        pred_enc_img = self.class_head_encoder_img(img_embedding)
        
        mask_embedding = self.encoder_mask(x["segmentation_mask"])
        pred_enc_mask = self.class_head_encoder_mask(mask_embedding)
        plane_embedding = torch.cat((img_embedding, mask_embedding),-1)
        
        pxs = x["ps"]
        pxs_embed = self.pixel_spacing_encoder(pxs)
        
        if self.landmarks:
            land = x["landmarks"]
            landmarks_embed = self.landmarks_encoder(land)
            pred_enc_landmarks = self.class_head_landmarks(landmarks_embed)
            plane_embedding = torch.cat((plane_embedding, landmarks_embed),-1)
        
        inputs = [plane_embedding, pxs_embed]

        dec_out = self.decoder(inputs)

        return img_embedding, mask_embedding, pred_enc_img, pred_enc_mask, pred_enc_landmarks, self.decoder.output_to_dict(dec_out)


class PretermFinetuning(pl.LightningModule):
    """Finetune Ultradino model"""

    ENCODER_DISABLE_GRAD = ["mask_token"]

    def __init__(
        self,
        model,
        freeze_encoder=True,
        use_proxy=True,
        # Optimizer hyperparameters
        optimizer_type="adamw",
        learning_rate=1e-4,
        weight_decay=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        # Scheduler hyperparameters
        scheduler_type="cosine_warmup",
        patience=5,
        factor=0.5,
        warmup_epochs=0,
        # Learning rate decay parameters
        lr_decay_rate=0.65,
        patch_embed_lr_mult=0.2,
    ):
        super().__init__()

        # Store hyperparameters for optimizer and scheduler
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        self.scheduler_type = scheduler_type
        self.patience = patience
        self.factor = factor
        self.warmup_epochs = warmup_epochs
        self.cosine_num_cycles = 0.5
        self.lr_decay_rate = lr_decay_rate
        self.patch_embed_lr_mult = patch_embed_lr_mult

        self.use_proxy = use_proxy
        self.freeze_encoder = freeze_encoder

        self.model = model

        # NB: At this point the keys are *not* prefixed with 'encoder.'!
        for n, p in self.model.encoder_img.named_parameters():
            if freeze_encoder or n in self.ENCODER_DISABLE_GRAD:
                p.requires_grad = False
        for n, p in self.model.encoder_mask.named_parameters():
            if freeze_encoder or n in self.ENCODER_DISABLE_GRAD:
                p.requires_grad = False
                
                
        # initialized during setup
        self.tb_logger = None
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        
    def barlow_like_loss(self,x, y):
        # normalize each dimension to zero mean and unit variance across batch
        x = (x - x.mean(0)) / (x.std(0) + 1e-4)
        y = (y - y.mean(0)) / (y.std(0) + 1e-4)
        B = x.size(0)
        C = (x.T @ y) / B  # (D,D)
        off_diag = C - torch.diag(torch.diagonal(C))
        return (off_diag**2).sum()
        
    def weightedRE(self, yh, y):
        """calculate the weighted relative error"""
        # mask out nan and 0 in y
        # mask = (y!=0) & (~torch.isnan(y))
        divy = 1 / y
        divy[torch.isinf(divy)] = 0
        return torch.mean((torch.abs(yh - y) * divy * (0.8)))#5 + torch.abs(weight))))
    
    def classification_loss(self, logit, label):
        # standard binary cross entropy loss
        return self.bce_loss(logit, label)
    
    def forward(self, x):
        pred_dict = self.model(x)

        return pred_dict

    def _process_encoder_param_groups(self, encoder_param_groups):
        """Process encoder parameter groups with layer-wise learning rates."""
        processed_groups = []

        for group in encoder_param_groups:
        
            lr_multiplier = group.get("lr_multiplier", 1.0)
            wd_multiplier = group.get("wd_multiplier", 1.0)

            if isinstance(lr_multiplier, (int, float)):
                actual_lr = self.learning_rate * lr_multiplier
            else:
                actual_lr = self.learning_rate

            if isinstance(wd_multiplier, (int, float)):
                actual_wd = self.weight_decay * wd_multiplier
            else:
                actual_wd = self.weight_decay

            processed_groups.append(
                {
                    "params": group["params"],
                    "lr": actual_lr,
                    "weight_decay": actual_wd,
                }
            )

        return processed_groups

    def _log_learning_rate_summary(self, param_groups):
        """Log a summary of learning rates for debugging."""
        logger.info("=== Layer-wise Learning Rate Summary ===")
        logger.info(
            f"Base LR: {self.learning_rate}, Decay: {self.lr_decay_rate}, "
            f"Patch mult: {self.patch_embed_lr_mult}"
        )

        # Group by learning rate and show summary
        lr_summary = {}
        for i, group in enumerate(param_groups):
            lr = group["lr"]
            if lr not in lr_summary:
                lr_summary[lr] = {"count": 0, "groups": []}
            # Convert generator to list to get count
            param_count = len(list(group["params"]))
            lr_summary[lr]["count"] += param_count
            lr_summary[lr]["groups"].append(i)

        for lr in sorted(lr_summary.keys()):
            info = lr_summary[lr]
            logger.info(f"LR {lr:.8f}: {info['count']} params, groups {info['groups']}")

    def configure_optimizers(self):
    
        param_groups = []

        # Get and process encoder parameter groups
        encoder_param_groups = get_param_groups_with_decay(
            model=self.model.encoder_img,
            lr_decay_rate=self.lr_decay_rate,
            patch_embed_lr_mult=self.patch_embed_lr_mult,
        )

        param_groups.extend(self._process_encoder_param_groups(encoder_param_groups))
        
                # Get and process encoder parameter groups
        encoder_param_groups = get_param_groups_with_decay(
            model=self.model.encoder_mask,
            lr_decay_rate=self.lr_decay_rate,
            patch_embed_lr_mult=self.patch_embed_lr_mult,
        )

        param_groups.extend(self._process_encoder_param_groups(encoder_param_groups))
        

        # Add decoder parameters
        param_groups.append(
            {
                "params": list(self.model.decoder.parameters()),
                "lr": self.learning_rate,
                "is_decoder": True,
            }
        )

        # Add pixel spacing embedding parameters if used
        if self.model.has_pixel_spacing_encoder():
            param_groups.append(
                {"params": list(self.model.pixel_spacing_encoder.parameters()), "lr": self.learning_rate}
            )
        # Add landmarks embedding parameters if used
        if self.model.has_landmarks_encoder():
            param_groups.append(
                {"params": list(self.model.landmarks_encoder.parameters()), "lr": self.learning_rate}
            )
            
                # Log summary for debugging
        self._log_learning_rate_summary(param_groups)

        # Create optimizer
        optimizer = self._create_optimizer(param_groups)

        # Configure scheduler if specified
        return self._configure_scheduler(optimizer)

    def _create_optimizer(self, param_groups):
        """Create the optimizer based on the specified type."""
        base_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

        if self.optimizer_type.lower() == "adam":
            return torch.optim.Adam(
                param_groups,
                **base_kwargs,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_eps,
            )
        elif self.optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(
                param_groups,
                **base_kwargs,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_eps,
            )
        elif self.optimizer_type.lower() == "sgd":
            return torch.optim.SGD(
                param_groups,
                **base_kwargs,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def _configure_scheduler(self, optimizer):
        """Configure the learning rate scheduler."""
        if self.scheduler_type.lower() == "none":
            return optimizer
        elif self.scheduler_type.lower() == "cosine":
            max_epochs = getattr(self.trainer, "max_epochs", 100) or 100
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        elif self.scheduler_type.lower() == "cosine_warmup":
            max_epochs = getattr(self.trainer, "max_epochs", 100) or 100
            steps_per_epoch = getattr(self, "_steps_per_epoch", 100)
            total_steps = max_epochs * steps_per_epoch
            warmup_steps = self.warmup_epochs * steps_per_epoch

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=self.cosine_num_cycles,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        elif self.scheduler_type.lower() == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.factor, patience=self.patience
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

    def setup(self, stage=None):
        if stage == "fit":
            self.tb_logger = next(
                logger
                for logger in self.trainer.loggers
                if isinstance(logger, TensorBoardLogger)
            )  # get a reference to the tb logger

            # Calculate steps per epoch for cosine scheduler with warmup
            trainer_dm = getattr(self.trainer, "datamodule", None)
            if trainer_dm is not None:
                # Get the datamodule
                datamodule = trainer_dm
                if hasattr(datamodule, "train_dataloader"):
                    # Setup the datamodule to get the dataloader
                    datamodule.setup("fit")
                    train_loader = datamodule.train_dataloader()
                    self._steps_per_epoch = len(train_loader)
                else:
                    raise ValueError(
                        "Datamodule does not have a train_dataloader method."
                    )
            else:
                raise ValueError("Trainer does not have a datamodule set up.")



    # pylint: disable=unused-argument
    def training_step(self, batch, batch_idx):

        inputs = batch

        img_embedding, mask_embedding, pred_enc_img, pred_enc_mask, pred_enc_landmarks, outputs = self.model(inputs)
        
        #decorr_loss = self.barlow_like_loss(img_embedding, mask_embedding) / 200
        
        logit_final = outputs['preterm'].squeeze(-1)
        logit_img = pred_enc_img.squeeze(-1)
        logit_mask = pred_enc_mask.squeeze(-1)
        logit_landmarks = pred_enc_landmarks.squeeze(-1)
        
        label = inputs['label'].float()
        
        logs = {}
        
        
        if self.use_proxy:
             pred_CL = outputs['CL']
             CL = inputs['CL']
             loss = self.classification_loss(logit_final, label) + self.weightedRE(pred_CL, CL) + 0.2*(self.classification_loss(logit_img, label) + self.classification_loss(logit_mask, label) + self.classification_loss(logit_landmarks, label)) #+ 0.1*decorr_loss#, torch.Tensor([0.3]))
        else:
             loss = self.classification_loss(logit_final, label) + 0.2*(self.classification_loss(logit_img, label) + self.classification_loss(logit_mask, label) + self.classification_loss(logit_landmarks, label)) #+ 0.1*decorr_loss
             
        logs["loss"] = loss

        auc = auroc(logit_final, label.long(), task='binary')
        spe = specificity(logit_final, label.long(), task='binary', threshold=0.5)
        rec = recall(logit_final, label.long(), task='binary', threshold=0.5)
        acc = accuracy(logit_final, label.long(), task='binary', threshold=0.5)

        logs["AUROC"] = auc
        logs["ACC"] = acc
        logs["SEN"] = rec
        logs["SPE"] = spe

        
        for key, value in logs.items():
            self.log(f"train_{key}", value, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        #self.log(
        #    "train_loss",
        #    loss,
        #    on_step=True,
        #    on_epoch=True,
        #    prog_bar=True,
        #    sync_dist=True,
        #)

        # Log learning rates
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        if hasattr(optimizer, "param_groups"):
            # Log base learning rate (first group, typically encoder parameters)
            base_lr = optimizer.param_groups[0]["lr"]
            self.log("lr_base", base_lr, on_step=True, on_epoch=False, sync_dist=True)

            # Find encoder groups (exclude decoder and pixel spacing groups)
            encoder_groups = []
            for i, group in enumerate(optimizer.param_groups):
                # Skip decoder group
                if group.get("is_decoder", False):
                    continue
                # Skip pixel spacing group (last group if pixel spacing is used)
                if (
                    self.model.has_pixel_spacing_encoder()
                    and i == len(optimizer.param_groups) - 1
                    and "is_decoder" not in group
                ):
                    continue
                # TODO: This probably needs to be adapted for the newly
                #       introduced gestational age encoder.
                encoder_groups.append(group)

            if encoder_groups:
                # Log the lowest LR among encoder groups (first layer)
                lowest_lr_group = min(encoder_groups, key=lambda g: g["lr"])
                lowest_lr = lowest_lr_group["lr"]
                self.log(
                    "lr_encoder_first",
                    lowest_lr,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
                print(f"Encoder first layer learning rate: {lowest_lr}")
            else:
                print("No encoder parameter groups found!")

        if (
            batch_idx % 1000 == 0 and self.trainer.is_global_zero
        ):  # avoid DDP duplicate logs
            if self.tb_logger is not None:
                tb_writer = self.tb_logger.experiment  # this is a SummaryWriter

                imgs = inputs["image"]
                tb_writer.add_images(
                        "input_images",
                        imgs.detach().cpu(),  # move to CPU; TB handles NCHW
                        global_step=self.global_step,
                    )

        return loss

    # pylint: disable=unused-argument
    def validation_step(self, batch, batch_idx):

        inputs = batch

        img_embedding, mask_embedding, pred_enc_img, pred_enc_mask, pred_enc_landmarks, outputs = self.model(inputs)
        
        logit_final = outputs['preterm'].squeeze(-1)
               
        label = inputs['label'].float()        
        
        logs = {}
        val_loss = self.classification_loss(logit_final, label)
        logs["loss"] = val_loss

        auc = auroc(logit_final, label.long(), task='binary')
        spe = specificity(logit_final, label.long(), task='binary', threshold=0.5)
        rec = recall(logit_final, label.long(), task='binary', threshold=0.5)
        acc = accuracy(logit_final, label.long(), task='binary', threshold=0.5)

        logs["AUROC"] = auc
        logs["ACC"] = acc
        logs["SEN"] = rec
        logs["SPE"] = spe
    
        for key, value in logs.items():
            self.log(f"val_{key}", value, on_step=False, on_epoch=True, sync_dist=True)

        #self.log("val_loss", val_loss,
        #    on_step=False,
        #    on_epoch=True,
        #    prog_bar=True,
        #    sync_dist=True,
        #)

        return val_loss
