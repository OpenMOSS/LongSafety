from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def get_model_name(self) -> str: ...
    
    @abstractmethod
    def format_inputs(self, inputs: str):
        """format the prompt to be fed into the model.
        For example, format the prompt into message format that the model can understand.

        Args:
            inputs (str): prompt to be formatted
        """
        ...
    
    
    @abstractmethod
    def generate_response(self, inputs, **kwargs) -> str: ...
    
    def truncate(self, text: str, max_length: int) -> str:
        """Truncate the text to fit the max_length from the middle

        Args:
            text (str): text to be truncated
            max_length (int): maximum length of the text

        Returns:
            str: truncated text
        """
        tokenizer = self.tokenizer
        tokens = tokenizer.encode(text)
        if len(tokens) > max_length:
            tokens = tokens[:int(max_length/2)] + tokens[-int(max_length/2):]
        return tokenizer.decode(tokens)