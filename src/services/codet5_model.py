from src.config.config import max_token_length
from src.services.base_model import BaseModel
import os
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class Codet5Model(BaseModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def load_model(self, model_name):
        model_path = os.path.join("models", model_name)
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            # Đọc thông tin cấu hình adapter
            config = PeftConfig.from_pretrained(model_path)
            base_model_name = config.base_model_name_or_path

            # Load mô hình cơ sở
            model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,  # Đảm bảo dùng precision đầy đủ
                device_map="auto"
            )
            # Load adapter từ checkpoint với đủ thông tin
            model = PeftModel.from_pretrained(
                model,
                model_path,
                is_trainable=False,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            print("checkpoint loaded")
        else:
            model_path = os.path.join("Salesforce", model_name)
            # Load mô hình thông thường từ HuggingFace
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                device_map="cuda" if torch.cuda.is_available() else "cpu"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("huggingface loaded")
        return model, tokenizer

    def generate_from_prompt(self, prompt: str):
        inputs = self.tokenizer(prompt + "<SEP>", return_tensors="pt", max_length=512, truncation=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        self.model.eval()  # bật chế độ đánh giá

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=inputs["input_ids"],
                max_length=max_token_length,
                pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer,
                                                                    'pad_token_id') else self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.5,  # thấp để không quá loạn
                top_p=0.9,  # nucleus sampling, bỏ bớt tail tokens
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


