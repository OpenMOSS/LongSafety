from .generator import Model

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HF_Model(Model):
    def __init__(self, model: str, model_name: str, max_length: int):
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def format_inputs(self, inputs):
        # check format of message
        if 'chat' in self.get_model_name().lower() or 'instruct' in self.get_model_name().lower():
            message = [{'role': 'user', 'content': inputs}]
            try:
                inputs = self.tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )[0]
            except Exception as e:
                print('Warning: Could not apply chat template to input. Use input as is.')
                print('Original error:', e)
        return inputs
    
    def generate_response(self, inputs, temperature=0, top_p=1.0, **kwargs) -> str:

        inputs = self.tokenizer.encode([inputs], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_length=self.max_length,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response =self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response