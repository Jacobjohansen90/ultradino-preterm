import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch

from torch import nn
from torch.utils.data import DataLoader

from data.pretermbirth_dataset import PretermBirthDataset
from torchvision.transforms import v2 as transforms

from typing import Optional, Union, Tuple, List

FUS13M_MEAN = 0.1842924807
FUS13M_STD = 0.2187705424

class PretermDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Preterm Study dataset.

    Wraps the original PretermBirthDataset and provides all the functionality
    from the original create_dataloader functions with Lightning integration.
    """

    def __init__(
        self, 
        csv_dir:str ='/data/proto/sPTB-SA-SonoNet/metadata/ASMUS_MICCAI_dataset_splits.csv',
        split_index:str = 'fold_1',
        label:str = 'birth_before_week_37',
        input_type:str = 'image',
        resample:str = False,
        batch_size: int = 64,
        img_size:tuple = (224, 224), 
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs 
    ):
        
        super().__init__()

        self.csv_dir = csv_dir
        self.split_index = split_index
        self.label_name = label
        self.input_type = input_type
        self.resample = resample
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = 16
        
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        
        self.norm_mean = [FUS13M_MEAN] 
        self.norm_std = [FUS13M_STD] 
        
        # Initialize transforms
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup image transforms for training and validation/test."""

        # Define the Albumentations transforms
        if self.input_type in ("image", 'mask_img'):
        
            self.train_transform = A.Compose(
                [
                    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), p=0.5),
                    # A.Affine(
                    #    scale=(0.8, 1.2),
                    #    translate_percent=(0.2, 0.2),
                    #    rotate=(-30, 30),
                    #    shear=(-15, 15),
                    #    interpolation=1,  # cv2.INTER_LINEAR
                    #    mode=1,  # cv2.BORDER_REFLECT_101
                    #    fit_output=True,
                    #    p=0.5),
                    A.HorizontalFlip(p=0.5),
                    
                    A.Resize(height=self.img_size[0], width=self.img_size[1]),
                    A.ToGray(p=1.0, num_output_channels=1),
                    A.Normalize(mean=self.norm_mean, std=self.norm_std),
                    ToTensorV2(),
                ]
            )
            
            self.val_transform = A.Compose(
                [
                    A.Resize(height=self.img_size[0], width=self.img_size[1]),
                    A.ToGray(p=1.0, num_output_channels=1),
                    A.Normalize(mean=self.norm_mean, std=self.norm_std),
                    ToTensorV2(),
                ]
            )
            
        elif self.input_type == "mask_only":
        
            self.train_transform = A.Compose(
                [
                    A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), p=0.5),
                    # A.Affine(
                    #    scale=(0.8, 1.2),
                    #    translate_percent=(0.2, 0.2),
                    #    rotate=(-30, 30),
                    #    shear=(-15, 15),
                    #    interpolation=1,  # cv2.INTER_LINEAR
                    #    mode=1,  # cv2.BORDER_REFLECT_101
                    #    fit_output=True,
                    #    p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.Resize(height=self.img_size[0], width=self.img_size[1]),
                    #A.ToGray(p=1.0, num_output_channels=1),
                    ToTensorV2(),
                ]
            )
            self.val_transform = A.Compose(
                [
                    A.Resize(height=self.img_size[0], width=self.img_size[1]),
                   
                    ToTensorV2(),
                ]
            )
            
        

        # Test transform is same as validation
        self.test_transform = self.val_transform
        
        # Initialize datasets (will be populated in setup())
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Called by Lightning to set up datasets for training, validation, and testing.
        This will be called automatically when fitting or testing the model.
        """

        """Setup datasets using the new dataloader logic (train/val_perm)."""

        if stage in ("fit", "validate", None):
            # Setup training dataset
           
            self.train_dataset = PretermBirthDataset(
            csv_dir=self.csv_dir, 
            split_index=self.split_index, 
            transforms=self.train_transform, 
            label_name=self.label_name, 
            input_type=self.input_type,
            resample=self.resample,
            split='train')

            self.val_dataset = PretermBirthDataset(
            csv_dir=self.csv_dir, 
            split_index=self.split_index, 
            transforms=self.val_transform, 
            label_name=self.label_name,
            input_type=self.input_type,
            resample=self.resample,
            split='vali'
        )
            

        if stage == "test" or stage is None:
            # For test, try to use test.pickle if available, otherwise use val_perm
            
            self.test_dataset = PretermBirthDataset(
            csv_dir=self.csv_dir, split_index=self.split_index, 
            transforms=self.test_transform,
            label_name=self.label_name,
            input_type=self.input_type,
            resample=self.resample,
            split='test'
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            drop_last=True,  # Drop last incomplete batch for training
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return prediction dataloader (same as test)."""
        return self.test_dataloader()
    
    
