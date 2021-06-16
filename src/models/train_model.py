from src.models.model import SRCNN
from src.data.dataloader import DIV2KDataModule
from pytorch_lightning import Trainer


def train(trainer, div2k, model):
    trainer.fit(model=model, datamodule=div2k)


def test(trainer, div2k, model):
    trainer.test(model=model, datamodule=div2k)


if __name__ == "__main__":
    div2k = DIV2KDataModule()
    model = SRCNN()

    logger = None  # Make into wandb at some point
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
