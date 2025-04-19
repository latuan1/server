import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config.config import model_name
from src.utils.model_utils import load_model_by_type

class GenerationRequest(BaseModel):
    input: str

app = FastAPI()

model = None

@app.on_event("startup")
async def load_model():
    global model
    model = load_model_by_type(model_name)

@app.post("/generate")
async def generate(request: GenerationRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail="Input không được để trống")
    result = model.generate_from_prompt(request.input)
    return {"output": result}

if __name__ == "__main__":
    # Chạy server trên host 0.0.0.0 và port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)