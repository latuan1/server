from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.utils.model_utils import load_model_by_type

# Định nghĩa schema dữ liệu đầu vào cho API
class GenerationRequest(BaseModel):
    input: str

app = FastAPI()

model = None

# Load model và tokenizer khi server khởi chạy
@app.on_event("startup")
async def load_model():
    global model
    model = load_model_by_type("models/codet5")


# Endpoint để generate output từ input
@app.post("/predict")
async def predict(request: GenerationRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail="Input không được để trống")
    result = model.predict(request.input)
    return {"output": result}