import azureml.core
from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.conda_dependencies import CondaDependencies

print("Using azureml-core version", azureml.core.VERSION)

# Get workspace from local config file (download through azure portal)
ws = Workspace.from_config()

# Create environment
env = Environment.from_pip_requirements(name="myenv", file_path='requirements.txt')

# Set dependencies of env (could also have used a dependencies.txt file instead)
packages = CondaDependencies.create(conda_packages=['pip', 'pytorch','python==3.7', 'joblib'],
                                    pip_packages=['azureml-defaults', 'torchvision==0.7.0',
                                                  'pytorch_lightning', 'wandb'])
env.python.conda_dependencies = packages

# I have created both a CPU and GPU compute targets
compute_targets = ws.compute_targets

# Run script using the GPU target and env
config = ScriptRunConfig(
    compute_target = compute_targets['GPU'],
    source_directory = 'src',
    script = 'models/train_model.py',
    environment = env
)

# Create experiment and run config on it
experiment_name = "Train_SRCNN"
exp = Experiment(ws, experiment_name)
run = exp.submit(config)