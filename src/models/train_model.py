import os
import shutil

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.data.dataloader import DIV2KDataModule
from src.models.model import SRCNN


def train(trainer, div2k, model):
    trainer.fit(model=model, datamodule=div2k)


def test(trainer, div2k, model):
    trainer.test(model=model, datamodule=div2k)


if __name__ == "__main__":

    # Read api key from file
    # File is in gitignore to ensure that key isn't pushed
    use_wandb = True
    try:
        with open("wandb_api_key.txt", encoding='utf-8') as f:
            key = f.read()
    except Exception:
        print("wandb_api_key.txt file not found containing api key"
              "cannot use WandB...")
        use_wandb = False

    logger = None

    if (use_wandb):
        wandb.login(key=key)
        wandb.init(entity='MLOps14', project="DIV2K")
        logger = WandbLogger()

    # ACTUAL TRAINING AND TESTING
    div2k = DIV2KDataModule()
    model = SRCNN()

    trainer = Trainer(max_epochs=100, logger=logger, gpus=-1)

    train(trainer, div2k, model)
    test(trainer, div2k, model)

    if (use_wandb):
        wandb.finish()

        # Delete local wandb files
        print(os.path.abspath(os.getcwd()))
        shutil.rmtree('wandb')
