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
        self.client = OpenAI(max_retries=10, timeout=120.0)
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

class AzureOpenaiEngine():

    def __init__(self, llm_engine_name):
        self.client = AzureOpenAI(api_key="3c744295edda4f13bf0ef198ddb4d24c",
        api_version="2024-10-21",
        azure_endpoint="https://ist-apim-aoai.azure-api.net/load-balancing/gpt-4.1")
        self.llm_engine_name = llm_engine_name

    def respond(self, user_input, temperature, top_p):
        response = self.client.chat.completions.create(
            model=self.llm_engine_name,
            messages=user_input,
            temperature=temperature,
            top_p=top_p,
            max_tokens=2000,
            frequency_penalty=0,
            presence_penalty=0        )

        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
