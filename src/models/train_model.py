from src.models.model import SRCNN
from src.data.dataloader import DIV2KDataModule
from pytorch_lightning import Trainer
import wandb
from pytorch_lightning.loggers import WandbLogger
import os
import shutil

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
    
    # Read api key from file
    # File is in gitignore to ensure that key isn't pushed
    use_wandb = True
    try:        
        with open("wandb_api_key.txt", encoding = 'utf-8') as f:
            key = f.read()
    except:
        print("wandb_api_key.txt file not found containing api key, cannot use WandB...")
        use_wandb = False
    
    logger = None
      
    if (use_wandb):  
        wandb.login(key=key)
        wandb.init(entity='MLOps14', project="DIV2K")
        logger = WandbLogger()
    
    wandb.log({"Test2": 123})
    
    #### ACTUAL TRAINING AND TESTING
    div2k = DIV2KDataModule()
    model = SRCNN()

    trainer = Trainer(max_epochs=100, logger=logger, gpus=0)

    train(trainer, div2k, model)
    test(trainer, div2k, model)
    
    if (use_wandb):
        wandb.finish()
        
        # Delete local wandb files
        print(os.path.abspath(os.getcwd()))
        shutil.rmtree('wandb')
        