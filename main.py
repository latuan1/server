from copy import deepcopy
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.config.config import model_name
from src.utils.model_utils import load_model_by_type

app = FastAPI()

class TestDataRequestModel(BaseModel):
    fm: Optional[str] = Field(None, description="file/module name")
    fc: Optional[str] = Field(None, description="class name")
    c: Optional[List[str]] = Field(default_factory=list, description="constructors / classes / contexts")
    f: Optional[List[str]] = Field(default_factory=list, description="fields")
    m: Optional[List[str]] = Field(default_factory=list, description="methods")

    def normalized(self) -> "TestDataRequestModel":
        # Trả về bản sao với danh sách luôn khác None và là copy (để tránh side effects)
        data = self.dict()
        data['c'] = deepcopy(self.c) if self.c is not None else []
        data['f'] = deepcopy(self.f) if self.f is not None else []
        data['m'] = deepcopy(self.m) if self.m is not None else []
        return TestDataRequestModel(**data)

def make_prompt(request: TestDataRequestModel) -> "str":
    r = request.normalized()
    executed_fm = (r.fm or "").strip()
    fc_name = (r.fc or "").strip()
    executed_f = " ".join([s.strip() for s in (r.f or []) if s.strip()])
    constructors = " ".join([s.strip() for s in (r.c or []) if s.strip()])
    methods = " ".join([s.strip() for s in (r.m or []) if s.strip()])
    fm_name = executed_fm.split('(', 1)[0]
    prompt = "Generate a test data input to maximize branch coverage for focal method " + fm_name + "".join(
        ["/*FC*/", fc_name, "\n{", "/*FM*/ ", executed_fm, "/*F*/:", executed_f, "/*C*/",
         constructors,"/*STUB*/", "/*M*/", methods, "\n}"
         ])
    return prompt

model = None

@app.on_event("startup")
async def load_model():
    global model
    model = load_model_by_type(model_name)

@app.get("/generate")
async def generate(request: TestDataRequestModel):
    if not request.input:
        raise HTTPException(status_code=400, detail="Input không được để trống")
    prompt = make_prompt(request)
    result = model.generate_from_prompt(prompt)
    return {"keyValue": result, "prompt": prompt}

if __name__ == "__main__":
    # Chạy server trên host 0.0.0.0 và port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)