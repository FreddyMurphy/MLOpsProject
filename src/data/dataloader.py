import glob
from typing import Optional

import kornia
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor


class DIV2K(torch.utils.data.Dataset):
    def __init__(self,
                 img_dir,
                 lr_trans=None,
                 hr_trans=None,
                 scale_factor=0.5,
                 image_size=(256, 256)):

        self.img_dir = img_dir
        self.img_labels = glob.glob(self.img_dir + '/*.png')

        self.scale_factor = scale_factor

        self.lr_trans = lr_trans
        self.hr_trans = hr_trans
        self.image_size = image_size

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        with Image.open(self.img_labels[idx]) as img:
            hr = self.get_hr_transforms()(img)
            lr = self.get_lr_transforms()(img)

        return hr, lr

    # Used to get the low res and high res transforms
    def get_lr_transforms(self):
        """Returns HR to LR image transformations"""
        return Compose([
            ToTensor(),
            kornia.geometry.Resize(self.image_size, align_corners=False),
            kornia.geometry.Rescale(self.scale_factor)
        ])

    def get_hr_transforms(self):
        """Returns HR image transformations"""
        return Compose([
            ToTensor(),
            kornia.geometry.Resize(self.image_size, align_corners=False)
        ])


class DIV2KDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = '',
                 batch_size: int = 8,
                 num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if self.data_dir != '':
            self.data_dir = self.data_dir + '/'
        self.div2k_train = DIV2K(self.data_dir + 'data/raw/DIV2K_train_HR')
        self.div2k_val = DIV2K(self.data_dir + 'data/raw/DIV2K_valid_HR')

        self.div2k_val, self.div2k_test = random_split(self.div2k_val, [95, 5])

        print("Train size:", len(self.div2k_train))
        print("Val size:", len(self.div2k_val))
        print("Test size:", len(self.div2k_test))

    def train_dataloader(self):
        return DataLoader(self.div2k_train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.div2k_val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.div2k_test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
