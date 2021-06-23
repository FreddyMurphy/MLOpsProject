import os

from azureml.core import Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice

import sys

ws = Workspace.from_config()

best_model = None
min_loss = sys.float_info.max
for model in Model.list(ws):
    if (model.name == 'div2k_model'):
        loss = float(model.properties['val_loss'])
        if (loss < min_loss):
            min_loss = loss
            best_model = model

# Takes newest model
## model = ws.models['div2k_model']

# Takes best model
model = best_model

print("Deploying:", model.name, 'version:', model.version)

folder_name = 'azure/upscale_service'
experiment_folder = './' + folder_name
os.makedirs(experiment_folder, exist_ok=True)
print(folder_name, 'folder created.')

script_file = os.path.join(experiment_folder, "score_upscale.py")

with open('./requirements.txt', encoding='utf-8') as f:
    pip_packages = f.readlines()
pip_packages = [x.strip() for x in pip_packages[1:]]  # Skip first line

pip_packages.append('azureml-defaults')

# Set dependencies of env
# (could also have used a dependencies.txt file instead)
packages = CondaDependencies.create(
    conda_packages=['pip', 'pytorch', 'python==3.7', 'joblib'],
    pip_packages=pip_packages)

git = "git+https://github.com/FreddyMurphy/MLOpsProject@master#egg=src"
packages.add_pip_package(git)

env_file = os.path.join(experiment_folder, "div2k_evn.yml")
with open(env_file, "w") as f:
    f.write(packages.serialize_to_string())
print("Saved dependency info in", env_file)

inference_config = InferenceConfig(runtime="python",
                                   entry_script=script_file,
                                   conda_file=env_file)

deployment_config = AciWebservice.deploy_configuration(cpu_cores=1,
                                                       memory_gb=1)

service_name = "upscale-service"

service = Model.deploy(ws, service_name, [model], inference_config,
                       deployment_config)

service.wait_for_deployment(True)
print(service.get_logs())
print(service.state)

print(service.scoring_uri)
