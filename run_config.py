import azureml.core
from azureml.core import (Dataset, Workspace, Experiment,
                          Environment, Model, ScriptRunConfig)
from azureml.core.conda_dependencies import CondaDependencies

# Training arguments
EPOCHS = 1
LEARNING_RATE = 0.0001

print("Using azureml-core version", azureml.core.VERSION)


# Get workspace from local config file (download through azure portal)
ws = Workspace.from_config()

# Create environment
env = Environment(workspace=ws, name="pytorch_env")

with open('requirements.txt', encoding='utf-8') as f:
    pip_packages = f.readlines()
pip_packages = [x.strip() for x in pip_packages[1:]]  # Skip first line

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
dataset = Dataset.File.from_files(path=(datastore, 'datasets'))
dataset_input = dataset.as_mount()

# Define arguments for config
arguments = ['train', '-e', EPOCHS, '-lr', LEARNING_RATE,
             '--data_dir', dataset_input]

# If wandb api key is defined, then send value
# as argument to enable usage of wandb logger
try:
    with open('wandb_api_key.txt', encoding='utf-8') as f:
        wandb_api_key = f.read()
        if wandb_api_key == '':
            raise Exception()

        arguments += ['--wandb_api_key', wandb_api_key]
except Exception:
    print("No wandb api key found")

# Run script using the GPU target and env
config = ScriptRunConfig(compute_target=compute_targets['GPU'],
                         source_directory='.',
                         script='src/models/main.py',
                         environment=env,
                         arguments=arguments)

# Create experiment and run config on it
experiment_name = "Train_SRCNN"
exp = Experiment(ws, experiment_name)
run = exp.submit(config)

run.wait_for_completion(show_output=True)

# run.upload_file(name='outputs/div2k_model.pkl', path_or_stream='./div2k_model.pkl')

run.complete()

# Register the model
run.register_model(model_path='outputs/div2k_model.pkl',
                   model_name='div2k_model',
                   tags={'Training context': 'Inline Training'})

for model in Model.list(ws):
    print(model.name, 'version:', model.version)
