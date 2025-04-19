from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name: str):
        self.model, self.tokenizer = self.load_model(model_name=model_name)

    @abstractmethod
    def load_model(self, model_name: str):
        pass

    @abstractmethod
    def generate_from_prompt(self, sample: str):
        pass