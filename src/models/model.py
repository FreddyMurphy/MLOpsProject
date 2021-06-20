import torch
import torch_enhance
from kornia.losses import SSIMLoss
from pytorch_lightning import LightningModule
from torch_enhance import metrics


class SRCNN(LightningModule):
    def __init__(self, scaling=4, n_channels=3, lr=0.001, window_size=5):
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

        # metrics
        # Mean absolute error
        mae = metrics.mae(upscaled, high_res)

        # Peak-signal-noise ratio
        psnr = metrics.psnr(upscaled, high_res)

        # Logs
        self.log("train_loss", loss)
        self.log("train_mae", mae)
        self.log("train_psnr", psnr)

        # TODO: Accuracy

        return loss

    # Very much similar to training step
    def validation_step(self, val_batch, batch_idx):
        high_res, low_res = val_batch
        upscaled = self.model(low_res)

        loss = self.criterion(upscaled, high_res)

        # metrics

        # Mean absolute error
        mae = metrics.mae(upscaled, high_res)

        # Peak-signal-noise ratio
        psnr = metrics.psnr(upscaled, high_res)

        # Logs
        self.log("train_loss", loss)
        self.log("train_mae", mae)
        self.log("train_psnr", psnr)

        # TODO: Accuracy

        return loss

    # Very much similar to training step
    def test_step(self, test_batch, batch_idx):
        high_res, low_res = test_batch
        upscaled = self.model(low_res)

        loss = self.criterion(upscaled, high_res)

        # metrics
        # Mean absolute error
        mae = metrics.mae(upscaled, high_res)

        # Peak-signal-noise ratio
        psnr = metrics.psnr(upscaled, high_res)

        # Logs
        self.log("train_loss", loss)
        self.log("train_mae", mae)
        self.log("train_psnr", psnr)

        # TODO: Accuracy

        return loss
