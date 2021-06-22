import os
import random
import shutil

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.functional import Tensor

import src.models.predict_model as predictor
from src.data.dataloader import DIV2KDataModule
from src.models.model import SRCNN
from src.models.train_model import test, train


class Session(object):
    def __init__(self, config):
        self.final_loss = 0
        session_params = config.session
        train_params = config.training
        model_params = config.model
        print(train_params)

        torch.manual_seed(session_params["seed"])
        random.seed(session_params['seed'])
        np.random.seed(session_params['seed'])

        train_or_evaluate = getattr(self, session_params['command'])

        self.model = self.setup_model(session_params['load_models_from'],
                                      model_params['learning_rate'],
                                      model_params['optim'])

        if (train_params['data_dir'] == '.'):
            self.data_dir = get_original_cwd()
        else:
            self.data_dir = train_params['data_dir']

        self.div2k = DIV2KDataModule(data_dir=self.data_dir,
                                     batch_size=train_params['batch_size'])
        self.epochs = train_params['epochs']

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

        if (self.use_wandb and not session_params['command'] == 'evaluate'):
            print("KEY:", key)
            wandb.login(key=key)
            wandb.init(entity='MLOps14', project="DIV2K")
            self.logger = WandbLogger()
            wandb.log(
                {
                    'learning_rate': model_params['learning_rate'],
                    'seed': session_params['seed'],
                    'batch_size': train_params['batch_size']
                },
                step=0)

            run_name = model_params['optim']
            run_name += '-' + wandb.run.name.split('-')[-1]
            run_name += '-' + 'lr=' + str(model_params['learning_rate'])
            wandb.run.name = run_name
            wandb.run.save()

        # Init finished, start either train or validate!
        train_or_evaluate()

    def train(self):
        model_path = os.path.join(get_original_cwd(), 'models/')
        # We only need a custom checkpoint to save to correct directory
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=model_path,
            filename='div2k-{epoch:02d}-{val_loss:.3f}',
            save_top_k=1,
            mode='min')

        early_stop_callback = EarlyStopping(monitor='val_loss',
                                            min_delta=0.01,
                                            patience=3,
                                            verbose=False,
                                            mode='min')

        gpus = -1 if torch.cuda.is_available() else 0

        trainer = Trainer(max_epochs=self.epochs,
                          logger=self.logger,
                          gpus=gpus,
                          callbacks=[checkpoint_callback, early_stop_callback])

        train(trainer, self.div2k, self.model)

        # Needed for Optuna tuning
        self.final_loss = trainer.logged_metrics['val_loss']

        best_model_path = os.path.join(get_original_cwd(),
                                       checkpoint_callback.best_model_path)

        if (self.use_wandb):
            wandb.finish()

            # Delete local wandb files
            # print(os.path.abspath(os.getcwd()))
            shutil.rmtree('wandb')

        # Save the best trained model to a separate directory
        # to potentially deploy using Azure later.
        model_file = os.path.join(get_original_cwd(), 'outputs',
                                  'div2k_model.ckpt')
        os.makedirs(get_original_cwd() + '/outputs', exist_ok=True)
        shutil.copyfile(best_model_path, model_file)

    def evaluate(self):
        trainer = Trainer(max_epochs=self.epochs, logger=self.logger, gpus=-1)

        test(trainer, self.div2k, self.model)
        predictor.save_model_output_figs(self.model)

    def setup_model(self, path, learning_rate, optimizer):
        model = SRCNN(lr=learning_rate, optimizer=optimizer)
        # Load model from checkpoint in case it was specificed
        if path != 'None':
            path = os.path.join(get_original_cwd(), path)
            model = SRCNN.load_from_checkpoint(path)
        return model


@hydra.main(config_path="../hparams", config_name="default_config")
def objective(config: DictConfig) -> Tensor:
    sess = Session(config)
    return sess.final_loss


if __name__ == '__main__':
    objective()
