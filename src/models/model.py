import torch
import torch_enhance
from torch import nn
import torch.nn.functional as F

class Module(nn.Module):

    def __init__(self, scaling, n_channels = 3, lr = 0.001) -> None:
        super().__init__()
        self.model = torch_enhance.models.SRCNN(scaling, n_channels)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = F.mse_loss() #####PUT criterion HERE######

    def forward(self, x):
        return self.model(x)


    #TODO: When we use pytorch lightning, we will need a method configure_optimizers(self):


    #TODO: To Cuda, maybe later
    def training_step(self, train_batch, batch_idx):

        #Model to training mode
        self.model.train()

        #Zero the gradients
        self.optimizer.zero_grad()


        low_res_i, high_res_i = train_batch
        upscaled_i = self.model(low_res_i)


        loss = self.criterion(upscaled_i, high_res_i)

        #Take a step
        loss.backward()
        self.optimizer.step()

        #TODO: Put logging stuff here...


        #TODO: We can return more stuff later, if necessary for logging
        return loss

    #TODO: To Cuda, sometime
    #Very much similar to training step
    def validation_step(self, val_batch, batch_idx):

        #Model to eval mode
        self.model.eval()

        #Zero the gradients
        self.optimizer.zero_grad()


        low_res_i, high_res_i = val_batch
        upscaled_i = self.model(low_res_i)


        loss = self.criterion(upscaled_i, high_res_i)


        #TODO: Put logging stuff here...


        #TODO: We can return more stuff later, if necessary for logging
        return loss

    #TODO: To Cuda, sometime
    #Very much similar to training step
    def test_step(self, test_batch, batch_idx):

        #Model to eval mode
        self.model.eval()

        #Zero the gradients
        self.optimizer.zero_grad()


        low_res_i, high_res_i = test_batch
        upscaled_i = self.model(low_res_i)


        loss = self.criterion(upscaled_i, high_res_i)


        #TODO: Put logging stuff here...


        #TODO: We can return more stuff later, if necessary for logging
        return loss