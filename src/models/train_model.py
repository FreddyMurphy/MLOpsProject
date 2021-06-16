from src.models.model import SRCNN
from src.data.dataloader import DIV2KDataModule
from pytorch_lightning import Trainer


'''
def train(model, device, trainloader, epochs=5, print_every=500):
    steps = 0
    running_loss = 0
    model.train()
    for e in range(epochs):
        # Model in training mode, dropout is on
        batch_idx = 0
        for batch_idx, images in trainloader:
            steps += 1
            images = images.to(device)

            loss = model.training_step(images, batch_idx)
            running_loss += loss.item()

            batch_idx += 1

            if steps % print_every == 0:

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. "
                      .format(running_loss/print_every))
                running_loss = 0
'''     

def train(trainer, div2k, model):
    trainer.fit(model=model, datamodule=div2k)
    

def test(trainer, div2k, model):
    trainer.test(model=model, datamodule=div2k)

if __name__ == "__main__":
    div2k = DIV2KDataModule()
    model = SRCNN()

    logger = None # Make into wandb at some point
    '''
    wandb.login(key='API KEY HERE')
    wandb.init(project="MNIST")
    
    logger = WandbLogger()
    '''
    trainer = Trainer(max_epochs=100, logger=logger, gpus=1)

    train(trainer, div2k, model)
    test(trainer, div2k, model)
    
    '''
    wand.finish()
    '''