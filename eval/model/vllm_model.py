from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from .generator import Model

class VLLM(Model):
    def __init__(self,  model: str, model_name: str, tensor_parallel_size: int, max_length):
        self.model = LLM(model, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True, max_seq_len_to_capture=max_length)
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.tokenizer.model_max_length = self.max_length
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def format_inputs(self, inputs: str):
        if 'chat' in self.get_model_name().lower() or 'instruct' in self.get_model_name().lower():
            inputs = [{'role': 'user', 'content': inputs}]
        return inputs
    
    def generate_response(self, inputs, temperature=0, top_p=1.0, **kwargs) -> str:
        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, **kwargs)
        if 'chat' in self.get_model_name().lower() or 'instruct' in self.get_model_name().lower():
            resp = self.model.chat(inputs, sampling_params=sampling_params)[0].outputs[0].text
        else:
            resp = self.model.generate([inputs], sampling_params=sampling_params)[0].outputs[0].text
            
        return resp
    
    def generate_choice_by_logits(self, inputs, candidates, temperature=0, top_p=1.0, logprobs=20, **kwargs):
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, logprobs=logprobs, **kwargs)
        if 'chat' in self.get_model_name().lower() or 'instruct' in self.get_model_name().lower():
            probs = self.model.chat(inputs, sampling_params=sampling_params)[0].outputs[0].logprobs[0]
        else:
            probs = self.model.generate([inputs], sampling_params=sampling_params)[0].outputs[0].logprobs[0]
        # check if the candidates are in the logprobs
        valid_candidates = []
        for v in probs.values():
            if v.decoded_token in candidates:
                valid_candidates.append(v)
        # return the candidate with the highest logprob
        resp = max(valid_candidates, key=lambda x: x.logprob)
        return resp.decoded_token
