# Human In The Loop Application.

This is a simple application that demonstrates how to use the ChainLit API with G.AI.T to create
a human-in-the-loop application to interact with the Find Existing Location (FEL) layers.

## Install Dependencies.

Make sure to install the dependencies by running:

```bash
uv pip install chainlit
```

## Configure.

Create a `.env` file and make sure to add the `LAYERS_JSON` key with a value that points to the layer JSON file.
Here is an example:

```dotenv
LAYERS_JSON=~/data/NorthSea.json
```

If you want to use S3, you can set the `LAYERS_JSON` to the S3 URL of the JSON file, like this:

```dotenv
AWS_ACCESS_KEY_ID=xxxx
AWS_SECRET_ACCESS_KEY=xxxx
# AWS_ENDPOINT_URL=http://localhost:9000 # Uncomment if using a local S3-compatible service like MinIO.
LAYERS_JSON=s3://my-bucket/path/to/NorthSea.json
```

## Run the application

```bash
chainlit run co_scientist.py --host 0.0.0.0 --port 8080
```

Note: You can add `-w` to watch for file changes.

## References

- https://github.com/Chainlit/cookbook
