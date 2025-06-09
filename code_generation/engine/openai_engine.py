from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, AzureOpenAI

import backoff

@backoff.on_exception(backoff.expo, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError))
def openai_chat_engine(client, engine, msg, temperature, top_p):
    if engine.startswith("gpt"):
        response = client.chat.completions.create(
            model=engine,
            messages=msg,
            temperature=temperature,
            max_tokens=2000,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0
        )
    else:
        response = client.chat.completions.create(
            model=engine,
            messages=msg,
        )

    return response

class OpenaiEngine():

    def __init__(self, llm_engine_name):
        # self.client = OpenAI(max_retries=10, timeout=120.0)
        self.client = AzureOpenAI(
            api_key='192cd313f4bb4a31b0fbba21a18f8c1a',
            api_version="2024-10-01-preview",
            azure_endpoint="https://seai.openai.azure.com/"
        )
        self.llm_engine_name = llm_engine_name

    def respond(self, user_input, temperature, top_p):
        response = openai_chat_engine(
            self.client, 
            self.llm_engine_name, 
            user_input,
            temperature,
            top_p
        )

        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens

class MyAzureOpenaiEngine:
    def __init__(self, api_key, api_version, azure_endpoint, model_name):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        self.model_name = model_name

    def respond(self, messages, temperature=0.7, top_p=0.95):

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=2000,
            frequency_penalty=0,
            presence_penalty=0
        )

        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
