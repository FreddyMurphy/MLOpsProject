import azureml.core
from azureml.core import Dataset, Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
print("Using azureml-core version", azureml.core.VERSION)

# Get workspace from local config file (download through azure portal)
ws = Workspace.from_config()

# Create environment
env = Environment(workspace=ws, name="pytorch_env")

# Set dependencies of env (could also have used a dependencies.txt file instead)
packages = CondaDependencies.create(conda_packages=['pip', 'pytorch','python==3.7', 'joblib'],
                                    pip_packages=['azureml-defaults', 'torchvision==0.7.0',
                                                  'pytorch_lightning', 'wandb', 'kornia', 'torch_enhance'])

env.python.conda_dependencies = packages
# env.python.conda_dependencies.add_conda_package('pip')
# env.python.conda_dependencies.add_conda_package('pytorch')
# env.python.conda_dependencies.add_conda_package('python==3.7')
# env.python.conda_dependencies.add_conda_package('joblib')

# I have created both a CPU and GPU compute targets
compute_targets = ws.compute_targets


datastore = ws.get_default_datastore()
datastore.upload(src_dir='./data',
                 target_path='datasets/data',
                 overwrite=False)

dataset = Dataset.File.from_files(path = (datastore, 'datasets'))
dataset_input = dataset.as_mount()
# Run script using the GPU target and env
config = ScriptRunConfig(
    compute_target = compute_targets['GPU'],
    source_directory = '.',
    script = 'src/models/train_model.py',
    environment = env,
    arguments=[dataset_input]
)

# Create experiment and run config on it
experiment_name = "Train_SRCNN"
exp = Experiment(ws, experiment_name)
run = exp.submit(config)
run.wait_for_completion(show_output=True)