import json

import kornia
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from azureml.core import Webservice, Workspace
from PIL import Image
from torchvision.transforms import Compose, ToTensor

ws = Workspace.from_config()

uri = ''

# Get endpoint uri for service
for s in Webservice.list(ws):
    if (s.name == 'upscale-service'):
        uri = s.scoring_uri

# If uri not found, then service is probably offline
# exit and perhaps try again
if (uri == ''):
    print("Did not find service...")
    exit()

# Load image, resize to our system and turn into tensor
with Image.open('./data/raw/DIV2K_valid_HR/0801.png') as img:
    transform = Compose([
        ToTensor(),
        kornia.geometry.Resize((256, 256), align_corners=False),
        kornia.geometry.Rescale(0.25)
    ])
    img = transform(img).unsqueeze_(0)

# Turn into numpy
img_numpy = img.numpy()

# Turn into list to be send as json
img_list = img_numpy.tolist()

# Create json dump for request
data = json.dumps({"data": img_list})
headers = {'Content-Type': 'application/json'}

# Create request
output = requests.post(uri, data, headers=headers)

# Look at result and turn into tensor
result = np.array(json.loads(output.json()))
result = torch.from_numpy(result)
print(result.shape)

plt.imshow(result.squeeze(0).permute(1, 2, 0))
plt.axis('off')
plt.show()
