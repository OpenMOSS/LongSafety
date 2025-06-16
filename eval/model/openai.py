from .generator import Model

from openai import OpenAI
import tiktoken
from transformers import AutoTokenizer

class OAIModel(Model):
    def __init__(self,  model: str, model_name: str, api_key: str=None, base_url: str=None, organization=None, max_length=128000):
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self.model = model
        self.model_name = model_name
        self.max_length = max_length
        self.client = OpenAI(api_key=api_key, base_url=base_url, organization=organization)
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            except:
                self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def format_inputs(self, inputs: str):
        return [{'role': 'user', 'content': inputs}]
    
    def generate_response(self, inputs, temperature=0, top_p=1.0, **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=inputs,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        ).choices[0].message.content
        return resp

    def generate_choice_by_logits(self, inputs, candidates, temperature=0, top_p=1.0, top_logprobs=5, max_tokens=1, **kwargs):
        probs = self.client.chat.completions.create(
            model=self.model,
            messages=inputs,
            temperature=temperature,
            top_p=top_p,
            logprobs=True,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            **kwargs
        ).choices[0].logprobs.content[0].top_logprobs
        valid_candidates = []
        chosen = None
        for v in probs:
            if v.token in candidates:
                valid_candidates.append(v)
        # return the candidate with the highest logprob
        if len(valid_candidates) > 0:
            chosen = max(valid_candidates, key=lambda x: x.logprob)
            chosen = chosen.token
        return chosen