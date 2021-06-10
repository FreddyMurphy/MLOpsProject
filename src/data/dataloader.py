
import torch
from torchvision.transforms import Compose, ToTensor, Resize
import torchvision.transforms as T
import pandas as pd
from torchvision.io import read_image
from typing import Tuple


class DIV2K(torch.utils.data.Dataset):


    def __init__(self, img_dir, transform=None, target_transform=None):

        self.img_dir = img_dir

        self.lr_transform: T.Compose = None
        self.hr_transform: T.Compose = None

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:

        hr = read_image(self.img_dir)
        lr = hr.copy()

        #Put images in tensors here...
        if self.hr_transform:
            hr = self.hr_transform(hr)
        if self.lr_transform:
            lr = self.lr_transform(lr)

        return hr, lr

    #Used to get the low res and high res transforms
    #TODO: Maybe we can add low res transformations from kornia
    def get_lr_transforms(self):
        """Returns HR to LR image transformations"""
        return Compose(
            [
                Resize(
                    size=(
                        self.image_size // self.scale_factor,
                        self.image_size // self.scale_factor,
                    ),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                ToTensor(),
            ]
        )

    #TODO: Check if this function does what it should
    def get_hr_transforms(self):
        """Returns HR image transformations"""
        return Compose(
            [
                Resize(
                    (self.image_size, self.image_size),
                    T.InterpolationMode.BICUBIC,
                ),
                ToTensor(),
            ]
        )
