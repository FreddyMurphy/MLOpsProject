import torch
import torch_enhance
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from kornia.losses import SSIMLoss


class SRCNN(LightningModule):

    def __init__(self, scaling=4, n_channels=3, lr=0.001, window_size = 5) -> None:
        super().__init__()
        self.model = torch_enhance.models.SRCNN(scaling, n_channels)
        
        self.lr = lr

        self.criterion = SSIMLoss(window_size)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, train_batch, batch_idx):
        high_res, low_res = train_batch
        upscaled = self.model(low_res)

        loss = self.criterion(upscaled, high_res)
        
        self.log('train_loss', loss)
        
        # TODO: Accuracy

        return loss

    # Very much similar to training step
    def validation_step(self, val_batch, batch_idx):
        high_res, low_res = val_batch
        upscaled = self.model(low_res)

        loss = self.criterion(upscaled, high_res)

        self.log('val_loss', loss)
        
        # TODO: Accuracy

        return loss

    # Very much similar to training step
    def test_step(self, test_batch, batch_idx):
        high_res, low_res = test_batch
        upscaled = self.model(low_res)

        loss = self.criterion(upscaled, high_res)

        self.log('test_loss', loss)
        
        # TODO: Accuracy

        return loss
