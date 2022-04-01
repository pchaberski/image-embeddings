"""Pytorch Lightning module for H&M images."""


import pytorch_lightning as pl
from .hm_data import HMDataset
from torch.utils.data import DataLoader
import os


class HMDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers = 1,
        image_size=[224, 224],
        center=False,
        center_params={'mean': None, 'std': None}
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.center = center
        self.center_params = center_params
        self.train_valid_ratio = self._get_train_valid_ratio()

    def setup(self, stage='fit'):
        self.data_train = HMDataset(
            data_path=os.path.join(self.data_path, 'train'),
            image_size=self.image_size,
            center=self.center,
            center_params=self.center_params
        )

        self.data_valid = HMDataset(
            data_path=os.path.join(self.data_path, 'valid'),
            image_size=self.image_size,
            center=self.center,
            center_params=self.center_params
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )

        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.data_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )

        return loader

    def _get_train_valid_ratio(self):
        data_path_train = os.path.join(self.data_path, 'train')
        data_path_valid = os.path.join(self.data_path, 'valid')

        jpg_fnames_train = [file for file in os.listdir(data_path_train) if file.endswith('.jpg')]
        jpg_fnames_valid = [file for file in os.listdir(data_path_valid) if file.endswith('.jpg')]

        return len(jpg_fnames_train)/(len(jpg_fnames_train) + len(jpg_fnames_valid))
