# Get workspace from local config file (download through azure portal)
from azureml.core import Webservice, Workspace

ws = Workspace.from_config()

# Delete all webservices connected to workspace
for i in Webservice.list(ws):
    print(i.name)
    i.delete()
