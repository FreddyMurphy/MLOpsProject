import os
import shutil

import hydra
import joblib
import torch
import wandb
from hydra.utils import get_original_cwd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import src.models.predict_model as predictor
from src.data.dataloader import DIV2KDataModule
from src.models.model import SRCNN
from src.models.train_model import test, train

import random
import numpy as np


class Session(object):
    def __init__(self, config):
        train_params = config.training
        # model_params = config.model

        torch.manual_seed(train_params['seed'])
        random.seed(train_params['seed'])
        np.random.seed(train_params['seed'])

        train_or_evaluate = getattr(self, train_params['command'])

        self.model = self.setup_model(train_params['load_models_from'],
                                      train_params['learning_rate'])

        if (train_params['data_dir'] == '.'):
            self.data_dir = get_original_cwd()
        else:
            self.data_dir = train_params['data_dir']

        self.div2k = DIV2KDataModule(data_dir=self.data_dir)
        self.epochs = train_params['epochs']
        self.learning_rate = train_params['learning_rate']

        # Try to find the wandb API key. The key can either be
        # passed as an argument (for use in Azure), or given in
        # the wandb_api_key.txt file.
        self.use_wandb = True
        try:
            with open(get_original_cwd() + "/wandb_api_key.txt",
                      encoding='utf-8') as f:
                key = f.read()
                if key == '':
                    raise Exception()
        except Exception:
            self.use_wandb = False

        if train_params['wandb_api_key'] != 'None':
            key = train_params['wandb_api_key']
            self.use_wandb = True

        if not self.use_wandb:
            print("wandb API key not found..." "Cannot use WandB...")

        self.logger = None

        if (self.use_wandb):
            print("KEY:", key)
            wandb.login(key=key)
            wandb.init(entity='MLOps14', project="DIV2K")
            self.logger = WandbLogger()

        # Init finished, start either train or validate!
        train_or_evaluate()

    def train(self):
        # We only need a custom checkpoint to save to correct directory
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='models/',
            filename='div2k-{epoch:02d}-{val_loss:.3f}',
            save_top_k=1,
            mode='min')

        gpus = -1 if torch.cuda.is_available() else 0

        trainer = Trainer(max_epochs=self.epochs,
                          logger=self.logger,
                          gpus=gpus,
                          callbacks=[checkpoint_callback])

        train(trainer, self.div2k, self.model)

        if (self.use_wandb):
            wandb.finish()

            # Delete local wandb files
            # print(os.path.abspath(os.getcwd()))
            shutil.rmtree('wandb')

        # Save the trained model
        model_file = 'div2k_model.pkl'
        os.makedirs(get_original_cwd() + '/outputs', exist_ok=True)
        joblib.dump(value=self.model.state_dict(),
                    filename=get_original_cwd() + '/outputs/' + model_file)

    def evaluate(self):
        trainer = Trainer(max_epochs=self.epochs, logger=self.logger, gpus=-1)

        test(trainer, self.div2k, self.model)
        predictor.save_model_output_figs(self.model)

    def setup_model(self, path, learning_rate):
        model = SRCNN(lr=learning_rate)
        # Load model from checkpoint in case it was specificed
        if path != 'None':
            model = SRCNN.load_from_checkpoint(path)
        return model


@hydra.main(config_path="../hparams", config_name="default_config")
def session(config):
    Session(config)


if __name__ == '__main__':
    session()
