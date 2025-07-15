# Generative AI Toolkit (G.AI.T)

<img src="media/GAIT2.png" width="256px" alt="G.AI.T Logo"/>

## Environment Variables

This project uses [LiteLLM](https://docs.litellm.ai/docs/providers) to interact with various LLM providers and models.
For example, if you are using Azure OpenAI, make sure to set the following environment variables:

```terminal
AZURE_API_BASE=https://xxxx.azure-api.net/load-balancing/gpt-4o-mini
AZURE_API_DEPLOYMENT=gpt-4o-mini
AZURE_API_KEY=xxxx
AZURE_API_VERSION=2024-10-21
```

Note that the above values are placeholders and should be replaced with your own values and are very specific to the ESRI Azure OpenAI API.

## Installation on MacOS or Linux

Make sure to install the latest version of [uv](https://docs.astral.sh/uv/getting-started/installation/), then execute the following commands:

```terminal
git clone https://github.com/EsriPS/gen-ai-toolkit
cd gen-ai-toolkit
uv venv --python=python3.11 --seed
source .venv/bin/activate
uv pip install -U .
```

**Note:** You can update uv using `uv self update` :-)

## Jupyter Lab

To use the Jupyter lab interface, you need to install the `jupyterlab` package. You can do this by running the following command:

```terminal
uv pip install ".[jupyter]"
```

Start a Jupyter lab session at port 8989 using for example the following command:

```terminal
jupyter lab --IdentityProvider.token="" --allow-root --no-browser --port 8989
```

**Note:** The above command does **NOT** have a token and is not secure. It is recommended to use a secure token for production environments.

## FastAPI

To use the FastAPI interface, you need to install the `fastapi` package. You can do this by running the following command:

```terminal
uv pip install ".[fastapi]"
```

Start a FastAPI server using for example the following command:

```terminal
cd fastapi
fastapi run main.py --app app
```

## Installation on Windows with ArcGIS Pro

Using a Python Command Prompt, execute the following commands:

```terminal
conda create --name gen-ai-toolkit --clone arcgispro-py3
proswap gen-ai-toolkit
cd %HOMEPATH%
git clone https://github.com/EsriPS/gen-ai-toolkit
cd gen-ai-toolkit
```

### For ArcGIS Pro 3.4

```terminal
pip install ".[arcgis34,fastapi]"
```

### For ArcGIS Pro 3.5

```terminal
pip install ".[arcgis35,fastapi]"
```

## References

- https://devtopia.esri.com/ArcGISPro/Python/wiki/Custom-Python-Registry-Keys-and-Environment-Variables
- https://github.com/BerriAI/litellm
- https://ollama.com
- https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
- https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
