from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def predict(self, prompt: str):
        pass