# Gen AI Toolkit in FastAPI

This is a sample example of how to use Gen AI Toolkit in a FastAPI application.

## Installation - Windows: 
- `conda create --yes --name gaitfastapi --clone arcgispro-py3`
- `cd gen-ai-toolkit`
- `pip install -U ".[arcgis]"`
- `pip install fastapi[standard]`

## Installtion - MacOS or Linux
- `uv pip install -U ".[fastapi]"`

## Usage: 
### Setup:
- Create a `.env` file with the following sample content
- `LAYERS_JSON` is a path to the output of Prepare Existing Locations tool. Ex: `C:\Projects\GAIT_Testing\GAIT_Testing.json`
```dotenv
TITLE="G.AI.T"
LAYERS_JSON=~/data/layers.json
AZURE_API_BASE=https://xxxx.azure-api.net/load-balancing/gpt-4.1
AZURE_API_DEPLOYMENT=gpt-4.1
AZURE_API_KEY=xxxx
AZURE_API_VERSION=2024-10-21
MODEL_NAME=azure/gpt-4.1
```

### Run commands:
Execute the following command to start the FastAPI server:
- Run local dev with auto reloading with: `fastapi dev main.py --app app`
- Run for production `fastapi run main.py --app app`
- or in background mode `nohup fastapi run main.py --app app &> fastapi.log`
- Open a browser and navigate to `http://localhost:8000/docs` to see the Swagger UI.
- You can stop the process by executing - `pkill -f fastapi`

## Docker Locally: 
**TODO: Improve this to use docker volume like Mansour had previously. I didn't see this at the time of creation and just used a `./layers` directory.**
- Copy *.json files from prepare existing locations output (Pro project path example: `C:\Users\dre11620\Documents\ArcGIS\Projects\GAIT_Testing`) to `fastapi/layers` directory
- Make sure there is a .env in the fastapi folder as outlined above in the setup section.
- `cd ..`
-`docker build -t gait-fastapi -f fastapi/Dockerfile .`
- `docker run -p 8000:8000 -d --name gait-fastapi-container gait-fastapi`
- Test locally first by using `Query Existing Locations` in the `GAITTools` toolbox. Point the tool to a localhost docker container

## Docker Deployment: 
- There are many azure container options https://learn.microsoft.com/en-us/azure/container-apps/compare-options
- Azure Container Instances are good for testing a simple container but are pay per second and don't set up a domain name/ssl certs by default 
- Azure Container Apps are serverless where they automatically spin up and down based on demand 
- Azure App Service seems the best fit - good for a simple API endpoint and allows you to setup domain name/ssl certs 
- This is the guide I followed https://bakshiharsh55.medium.com/deploy-python-fastapi-using-azure-container-registry-71c332f88ffb
- Another guide which uses azure cli https://learn.microsoft.com/en-us/azure/developer/python/tutorial-containerize-simple-web-app-for-app-service?tabs=web-app-fastapi

### Step 1 - Create ACR: 
- Azure Container Registry is a private Docker registry that stores images for use in Azure Container based applications 
- Create one by searching for "container registry" in Azure web portal 
- You can also use the azure cli to create one with `az acr create --resource-group web-app-simple-rg --name <container-registry-name> --sku Basic`
- I selected `Tenant Reuse` for Domain name label scope, read more here https://learn.microsoft.com/en-us/azure/container-registry/container-registry-get-started-portal?tabs=azure-cli. `Unsecure` is the default 
- I selected `Basic` for pricing plan 

### Step 2 - Access Key: 
- Grab access key from the ACR in `Settings->Access keys` in azure web portal 
- Enable the admin user and grab the username and password 
- Take note of the url 

### Step 3 - Docker CLI Login: 
- run `docker login [Your Azure Login server URL] -u [Username] -p [Password]` 
- Example: `docker login gaittestcontainers-cafbbmadg0ddbpas.azurecr.io -u GaitTestContainers -p mypassword`

### Step 4 - Docker build and push: 
- Note: Don't forgot to copy *.json files from prepare existing locations output (Pro project path example: `C:\Users\dre11620\Documents\ArcGIS\Projects\GAIT_Testing`) to `fastapi/layers` directory
- `docker build -t [Azure Login server]/[Container name]:[Tag] .`
- Example: `docker build -t gaittestcontainers-cafbbmadg0ddbpas.azurecr.io/gait-fastapi:v1 -f fastapi/Dockerfile .`
- `docker push [Azure Login server]/[Container name]:[Tag]` 
- Example: `docker push gaittestcontainers-cafbbmadg0ddbpas.azurecr.io/gait-fastapi:v1`
- Note: This will take some time, the GAIT image is 9.5GB
- Go to `Repositories` in your container registry in azure web portal and you should see your image there if it worked 

### Step 5 - Create App Service: 
- Go back to your resource group in azure web portal and select `Create` and search for `App Service` 
- Choose `Web App` 
- Choose `Container`, `Linux`
- Set your region, resource group, and instance name 
- Set your pricing plan (there should be a pricing breakdown table). I went with Premium which is 4GB of ram and ~40$ a month 
- Go to `Container` tab and select ACR for `Image Source`
- Select the ACR you created in step 1 and then select `Admin Credentials` or `Managed Idenity` (I chose Admin Credentials)
- Select your docker image you created in step 4
- Go to `Networking` tab and ensure `Enable Public Access` is on 
- Everything else I left default and do `Review and Create`
- After deployment select `Go to resource` and if everything worked you should be able to send requests to the `Default Domain` provided 
- It make take a few mins to spin up and you can check the `Logs` tab to make sure its working 

### Step 6 - GP Deployment: 
- Point the query existing locations GP Tool to the azure fastAPI server Ex: `https://gaitdemo-chfvf0fkgbhgajeg.westus2-01.azurewebsites.net/` and confirm it runs sucessfully in local ArcGIS Pro
- If it worked, go to GP history and right click on the successful run. Select `Share as Web tool` 
- Configure all options and set message level to `Info` to see more logging info 
- Analyze the tool to make sure all is good and then select `Publish` 
- Confirm it worked by navigating to `Portal` in Catalog pane and selecting your web tool 
- Run the web tool in Pro and confirm it works 
- Open AGOL 
- Open your hosted feature layer in map viewer 
- Click on `Analysis` on the right side menu
- Select `Browse Custom Web Tools` and choose `QueryExistingLocations`. Run the tool AGOL and confirm it works

## Use VS Code Python Debugger with FastAPI: 
- Open `main.py` of the FastAPI app
- Select the play button in the top right and choose `Python Debugger:Debug using launch.json`
- Create a launch.json that looks similar to this: 
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: FastAPI",
            "type": "debugpy",
            "request": "launch",
            "module": "fastapi",
            "args": [
                "dev",
                "main.py", 
                "--app", "app"
            ],
            "cwd": "${workspaceFolder}/fastapi",
            "jinja": true, 
            "justMyCode": false
        }
    ]
}
```
- Set the python interpreter to your conda env with gait and fastapi installed by doing `CTRL+SHIFT+P` and typing `Python:Select Interpreter` 
- Then select the play button in the top right and choose `Python Debugger:Debug using launch.json` which should open a debugger terminal 
- You should now be able to place breakpoints in the GAIT fastapi code

## TODO: Docker Volume
**TODO: Improve this to use docker volume like Mansour had previously. I didn't see this at the time of creation and just used a `./layers` directory.**
Make sure to adjust the volume path to your data directory:

```bash
docker run -it --rm --name gait-fastapi -p 8000:8000 -v ${HOME}/data:/data gait-fastapi
```

For better start up next time and to not download everytime the embedding weights, you can commit the container to an image:

```bash
docker commit <container-id> gait-fastapi:<your-tag>
```

You can find the `<container-id>` by running:

```bash
docker ps -a
```

Then you can run the container with:

```bash
docker run -it --rm --name gait-fastapi -p 8000:8000 -v ${HOME}/data:/data gait-fastapi:<your-tag>
```

## References
- https://medium.com/@rameshkannanyt0078/10-hidden-gem-libraries-to-supercharge-your-fastapi-projects-249f6decba05
- https://fastapi-mcp.tadata.com/getting-started/welcome
- https://huggingface.co/blog/lynn-mikami/fastapi-mcp-server
- https://bakshiharsh55.medium.com/deploy-python-fastapi-using-azure-container-registry-71c332f88ffb