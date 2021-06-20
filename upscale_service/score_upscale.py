import json
import pickle

import numpy as np
import torch
from azureml.core.model import Model

from src.models.model import SRCNN


def init():
    global model
    model_path = Model.get_model_path('div2k_model')
    model = SRCNN()

    with open(model_path, 'rb') as f:
        obj = f.read()
        state_dict = {
            key: arr
            for key, arr in pickle.loads(obj, encoding='latin1').items()
        }
    model.load_state_dict(state_dict)


def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])

    upscaled = model(torch.from_numpy(data).float())

    return json.dumps(upscaled.detach().numpy().tolist())
