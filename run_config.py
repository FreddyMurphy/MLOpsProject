import azureml.core
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
print("Using azureml-core version", azureml.core.VERSION)

# Get workspace from local config file (download through azure portal)
ws = Workspace.from_config()

# Create environment
env = Environment(workspace=ws, name="pytorch_env")

# Set dependencies of env (could also have used a dependencies.txt file instead)
packages = CondaDependencies('requirements.txt')
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
experiment_name = "Train SRCNN"
exp = Experiment(ws, experiment_name)
run = exp.submit(config)