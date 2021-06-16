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
        # Setup parsing of arguments
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>")
        parser.add_argument("command",
                            metavar='<command>',
                            help="Subcommand to run; train or validate")

        parser.add_argument('--epochs', '-e',
                            type=int,
                            metavar='<integer>',
                            help='Number of epochs to train',
                            default=10)

        parser.add_argument('--learning_rate', '-lr',
                            type=float,
                            metavar='<float>',
                            help='Learning rate during training',
                            default=0.0001)

        parser.add_argument('--load_models_from', '-l',
                            type=str,
                            metavar='<string>',
                            help='Model file path',
                            default=None)

        args = parser.parse_args()

        # Exit gracefully if wrong command is provided
        if not hasattr(self, args.command):
            print('Unkown command:', args.command)
            parser.print_help()
            exit(1)

        train_or_validate = getattr(self, args.command)

        self.model = self.setup_model(args.load_models_from,
                                      args.learning_rate)
        self.div2k = DIV2KDataModule()
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate

        # Init finished, start either train or validate!
        train_or_validate()

    def train(self):
        # We only need a custom checkpoint to save to correct directory
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='models/',
            filename='div2k-{epoch:02d}-{val_loss:.3f}',
            save_top_k=1,
            mode='min')

        logger = None  # Make into wandb at some point

        self.model.train()
        trainer = Trainer(max_epochs=self.epochs,
                          logger=logger,
                          gpus=1,
                          callbacks=[checkpoint_callback])

        train(trainer, self.div2k, self.model)

    def validate(self):
        logger = None  # Make into wandb at some point
        trainer = Trainer(max_epochs=self.epochs, logger=logger, gpus=1)

        self.model.eval()
        test(trainer, self.div2k, self.model)
        predictor.save_model_output_figs(self.model)

    def setup_model(self, path, learning_rate):
        model = SRCNN(lr=learning_rate)

        # Load model from checkpoint in case it was specificed
        if path:
            model = SRCNN.load_from_checkpoint(path)

        return model


if __name__ == '__main__':
    session = Session()
