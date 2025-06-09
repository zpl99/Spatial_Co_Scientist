from openai import AzureOpenAI

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
