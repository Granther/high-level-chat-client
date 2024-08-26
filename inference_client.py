from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
import os

class InferenceClient():
    def __init__(
        self,
        model: str=None,
        api_key: str=None,
        inference_url: str=None,
        max_new_tokens: int=3000,
        global_sys_prompt: str=None,
        backend: str="groq", 
        temperature: float=1.0,
        groq_key_name: str="GROQ_API_KEY",
        openai_key_name: str="OPENAI_API_KEY",
        verbose: bool=False
    ):
        load_dotenv()

        if not model:
            return ValueError("Must specify model") 

        self.backends = ['groq', 'openai']
        self.api_key = api_key
        self.model = model
        self.inference_url = inference_url
        self.max_new_tokens = max_new_tokens
        self.global_sys_prompt = global_sys_prompt
        self.backend = backend.lower()
        self.temperature = temperature

        if backend not in self.backends:
            raise ValueError(f"backend must be one of these: {self.backends}")
        
        if verbose:
            print(f"InferenceClient: Using {backend} backend")
            if not api_key:
                print(f"InferenceClient: Attempting to use API key from .env since api_key was not passed")
        
        if backend == "groq":
            self.client = Groq(api_key=api_key or os.getenv(groq_key_name))
        elif backend == "openai":
            self.client = OpenAI(api_key=api_key or os.getenv(openai_key_name), base_url=inference_url)


    def client_infer(self, prompt, sys_prompt:str=None, model:str=None, stream:bool=False, max_new_tokens:int=None, temperature:float=None):
        model = model or self.model
        sys_prompt = sys_prompt or self.global_sys_prompt
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "you are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Explain the importance of fast language models",
                }
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=1,
            stream=stream,
        )

        if stream is True:
            # Returns generator func
            return self.process_stream(response)

        elif stream is False:
            return response.choices[0].message.content
        
    def set_sys_prompt(self, new_sys):
        self.global_sys_prompt = new_sys

    def set_model(self, new_model):
        self.model = new_model

    # Function that is a generator
    def process_stream(self, stream):
        partial_message = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                partial_message = partial_message + chunk.choices[0].delta.content
                yield partial_message

if __name__ == "__main__":
    infer = InferenceClient(model="gemma2-9b-it", api_key="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", verbose=True)
    print(infer.client_infer("Heloe"))