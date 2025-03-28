from src.config.config import MODEL_NAMES
from src.services.codet5_model import Codet5Model

def load_model_by_type(model_name):
    if model_name == MODEL_NAMES.CODET5:
        return Codet5Model(model_name)
    elif model_name == MODEL_NAMES.STARCODER2_3B:
        return
