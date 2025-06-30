import os
import openai
import backoff
from prompt_toolkit.search import stop_search

completion_tokens = prompt_tokens = 0

# Azure OpenAI configs
api_key = os.getenv("AZURE_OPENAI_API_KEY", "3c744295edda4f13bf0ef198ddb4d24c")
api_base = os.getenv("AZURE_OPENAI_API_BASE", "https://ist-apim-aoai.azure-api.net/load-balancing/gpt-4.1")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "gpt-4.1")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "2024-10-21")

if api_key == "" or api_base == "" or api_version == "" or deployment == "":
    raise RuntimeError("Some required Azure OpenAI env vars are missing!")

client = openai.AzureOpenAI(api_key="3c744295edda4f13bf0ef198ddb4d24c",
        api_version="2024-10-21",
        azure_endpoint="https://ist-apim-aoai.azure-api.net/load-balancing/gpt-4.1")

@backoff.on_exception(backoff.expo, openai.OpenAIError)
def completions_with_backoff(messages,temperature, stop, n):
    # Azure API uses deployment_id instead of model
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=temperature,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
        stop = stop,
        n = n)
    return response

def gpt(prompt, model=deployment, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, temperature=temperature,  n=n, stop=stop)

def chatgpt(messages, temperature=0.7, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(
            messages=messages,
            temperature=temperature,
            n=cnt,
            stop=stop
        )
        outputs.extend([choice.message.content for choice in res.choices])
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs

def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    # 价格请根据你实际 Azure 定价改
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.00250 + prompt_tokens / 1000 * 0.01
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
