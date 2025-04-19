from src.services.base_model import BaseModel
import os
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class StarCoderModel(BaseModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def load_model(self, model_name):
        model_path = os.path.join("models", model_name)
        is_local_model = os.path.exists(model_path)
        is_peft_model = is_local_model and os.path.exists(os.path.join(model_path, "adapter_config.json"))

        if is_peft_model:
            # Trường hợp 2: Load AdaLoRA model đã fine-tune
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_name = peft_config.base_model_name_or_path

            # Load base model trước
            print(f"Loading base model: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )

            # Sau đó load PEFT adapter
            print(f"Loading PEFT adapter from: {model_name}")
            model = PeftModel.from_pretrained(base_model, model_path)

            # ===== BẮT ĐẦU CODE KIỂM TRA =====
            # In thông tin adapter config để kiểm tra
            print("\n--- ADAPTER CONFIGURATION ---")
            import json
            try:
                with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                    adapter_config = json.load(f)
                    print(json.dumps(adapter_config, indent=2))
            except Exception as e:
                print(f"Không thể đọc file adapter_config.json: {e}")

            # Kiểm tra các adapter module
            # Kiểm tra các adapter module
            print("\n--- ADAPTER MODULES CHECK ---")
            adapter_found = False
            for name, module in model.named_modules():
                # Kiểm tra xem module có phải là AdaLoRA module không
                if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                    adapter_found = True
                    print(f"Found adapter module: {name}")

                    # Cách truy cập đúng cho ParameterDict trong AdaLoRA
                    if hasattr(module, 'lora_A'):
                        # Trong AdaLoRA, lora_A là ParameterDict, cần kiểm tra keys
                        print(f"  lora_A keys: {list(module.lora_A.keys())}")
                        for key in module.lora_A.keys():
                            param = module.lora_A[key]
                            print(
                                f"  lora_A[{key}] shape: {param.shape}, non-zero: {torch.count_nonzero(param).item()}")

                    if hasattr(module, 'lora_B'):
                        print(f"  lora_B keys: {list(module.lora_B.keys())}")
                        for key in module.lora_B.keys():
                            param = module.lora_B[key]
                            print(
                                f"  lora_B[{key}] shape: {param.shape}, non-zero: {torch.count_nonzero(param).item()}")

            if not adapter_found:
                print("CẢNH BÁO: Không tìm thấy adapter LoRA trong mô hình!")
            else:
                print("THÀNH CÔNG: Đã tìm thấy adapter LoRA trong mô hình!")

            # Load tokenizer từ base model
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        else:
            model_path = os.path.join("bigcode", model_name)
            # Trường hợp 1: Load model gốc từ HuggingFace
            print(f"Loading model from HuggingFace: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cuda" if torch.cuda.is_available() else "cpu"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Đảm bảo có pad_token
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def generate_from_prompt(self, prompt: str):
        inputs = self.tokenizer(prompt + "<SEP> ", return_tensors="pt", max_length=512, truncation=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(inputs=inputs["input_ids"], max_length=512,
                                          pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)