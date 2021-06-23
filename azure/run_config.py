import sys

import azureml.core
import joblib
import torch
from azureml.core import (Dataset, Environment, Experiment, Model,
                          ScriptRunConfig, Workspace)
from azureml.core.conda_dependencies import CondaDependencies

from src.models.model import SRCNN

# Training arguments if one wants to override hydra config
# EPOCHS = 10
# LEARNING_RATE = 0.0001
# SEED = 1234

if __name__ == '__main__':
    print("Using azureml-core version", azureml.core.VERSION)

    # Get workspace from local config file (download through azure portal)
    ws = Workspace.from_config()

    # Create environment
    env = Environment(workspace=ws, name="pytorch_env")

    with open('./requirements.txt', encoding='utf-8') as f:
        pip_packages = f.readlines()
    pip_packages = [x.strip() for x in pip_packages[1:]]  # Skip first line

    pip_packages.append('azureml-defaults')

    # Set dependencies of env
    # (could also have used a dependencies.txt file instead)
    packages = CondaDependencies.create(
        conda_packages=['pip', 'pytorch', 'python==3.7', 'joblib'],
        pip_packages=pip_packages)

    env.python.conda_dependencies = packages

    # I have created both a CPU and GPU compute targets
    compute_targets = ws.compute_targets

    # Upload data to ws datastore if not present
    datastore = ws.get_default_datastore()

    datastore.upload(src_dir='./data',
                     target_path='datasets/data',
                     overwrite=False)

    # Find path of datastore and mount to compute
    path = (datastore, 'datasets')
    dataset = Dataset.File.from_files(path=path)
    data_ref = datastore.path('datasets').as_mount()
    dataset_input = dataset.as_mount()

    arguments = ['training.data_dir=' + str(data_ref)]

    # If wandb api key is defined, then send value
    # as argument to enable usage of wandb logger
    try:
        with open('./wandb_api_key.txt', encoding='utf-8') as f:
            wandb_api_key = f.read()
            if wandb_api_key == '':
                raise Exception()

            arguments += ['training.wandb_api_key=' + wandb_api_key]
    except Exception:
        print("No wandb api key found")

    arguments += sys.argv[1:]
    print("Starting run with following arguments:", arguments)
    # Run script using the GPU target and env
    config = ScriptRunConfig(compute_target=compute_targets['GPU'],
                             source_directory='.',
                             script='./src/models/main.py',
                             environment=env,
                             arguments=arguments)
    config.run_config.data_references = {
        data_ref.data_reference_name: data_ref.to_config()
    }

    # Create experiment and run config on it
    experiment_name = "Train_SRCNN"
    exp = Experiment(ws, experiment_name)
    run = exp.submit(config)

    run.wait_for_completion(show_output=True)

    run.complete()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Register the model
    run.download_file('outputs/div2k_model.ckpt', './outputs/div2k_model.ckpt')
    checkpoint = torch.load('./outputs/div2k_model.ckpt', map_location=DEVICE)

    val_loss = None
    for key in checkpoint['callbacks'].keys():
        try:
            val_loss = checkpoint['callbacks'][key]['best_model_score'].item()
        except Exception:
            pass

    model = SRCNN.load_from_checkpoint('./outputs/div2k_model.ckpt')
    # Save the state dict of the best trained model
    model_file = 'div2k_model.pkl'
    joblib.dump(value=model.state_dict(), filename='./outputs/' + model_file)
    run.upload_file('outputs/div2k_model.pkl', './outputs/div2k_model.pkl')

    run.register_model(model_path='./outputs/div2k_model.pkl',
                       model_name='div2k_model',
                       properties={
                           'lr': model.lr,
                           'optim': model.optimizer,
                           'val_loss': val_loss
                       })

    for model in Model.list(ws):
        print(model.name, 'version:', model.version)
