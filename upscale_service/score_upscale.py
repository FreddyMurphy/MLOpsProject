import json
import joblib
import numpy as np
import os
from azureml.core.model import Model


def init():
    global model
    
    print('azure_model_dir:', os.getenv('AZURE_MODEL_DIR'))
    model_path = Model.get_model_path('div2k_model')
    print('model_path:', model_path)
    model = joblib.load(model_path)
    
def run(raw_data):
    # data = np.array(json.loads(raw_data)['data'])
    # upscaled = model.predict(raw_data)
    
    return json.dumps([1,2,3])