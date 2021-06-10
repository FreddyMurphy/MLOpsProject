import torch
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms as T
from torchvision.io import read_image
from typing import Tuple
import kornia
import glob


class DIV2K(torch.utils.data.Dataset):
    def __init__(self,
                 img_dir,
                 transform=None,
                 target_transform=None,
                 scale_factor=0.25):

        self.img_dir = img_dir
        self.img_labels = glob.glob(self.img_dir + '/*.png')
        self.lr_transform: T.Compose = None
        self.hr_transform: T.Compose = None

        self.scale_factor = scale_factor

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        hr = read_image(self.img_labels[idx])
        lr = torch.clone(hr)

        return hr, lr

    # Used to get the low res and high res transforms
    def get_lr_transforms(self):
        """Returns HR to LR image transformations"""
        return Compose([
            kornia.geometry.Rescale(self.scale_factor),
            ToTensor(),
        ])

    # TODO: Check if this function does what it should
    def get_hr_transforms(self):
        """Returns HR image transformations"""
        return Compose([ToTensor()])


if __name__ == '__main__':
    data = DIV2K('../../data/raw/DIV2K_train_HR')

    train = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

    for highres, lowres in train:
        print('high:', highres)
        print('lowres:', lowres)
