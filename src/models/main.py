import sys
import argparse
from src.models.model import SRCNN
from src.data.dataloader import DIV2KDataModule
from src.models.train_model import train, test
import src.models.predict_model as predictor
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


class Session(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>")
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unkown command')

            parser.print_help()
            exit(1)
        train_or_validate = getattr(self, args.command)
        train_or_validate()

    def train(self):
        # Load data and model
        div2k = DIV2KDataModule()
        model = self.setup_model()

        # We only need a custom checkpoint to save to correct directory
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='models/',
            filename='div2k-{epoch:02d}-{val_loss:.3f}',
            save_top_k=1,
            mode='min')

        logger = None  # Make into wandb at some point

        model.train()
        trainer = Trainer(max_epochs=100,
                          logger=logger,
                          gpus=1,
                          callbacks=[checkpoint_callback])

        train(trainer, div2k, model)

    def validate(self):
        div2k = DIV2KDataModule()
        model = self.setup_model()

        logger = None  # Make into wandb at some point
        trainer = Trainer(max_epochs=100, logger=logger, gpus=1)

        model.eval()
        test(trainer, div2k, model)
        predictor.save_model_output_figs(model)

    def setup_model(self):
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="")
        args = parser.parse_args(sys.argv[2:])

        model = SRCNN()

        # Load model from checkpoint in case it was specificed
        if args.load_model_from:
            model = SRCNN.load_from_checkpoint(args.load_model_from)

        return model


if __name__ == '__main__':
    session = Session()
