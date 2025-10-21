import re
from copy import deepcopy
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from src.config.config import model_name, version_id, max_token_length
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


def extract_class_declaration(focal_class):
    match = re.search(r'\b(?:final\s+)?(?:class)\s+\w+\s*{', focal_class)

    if match:
        return match.group(0).strip(" {")
    else:
        return ""


def make_prompt(request: TestDataRequestModel) -> "str":
    r = request.normalized()
    executed_fm = (r.fm or "").strip()
    fc_name = extract_class_declaration(r.fc)
    executed_f = " ".join([s.strip() for s in (r.f or []) if s.strip()])
    constructors = " ".join([s.strip() for s in (r.c or []) if s.strip()])
    methods = " ".join([s.strip() for s in (r.m or []) if s.strip()])
    fm_name = executed_fm.split('(', 1)[0]
    prompt = "Generate a test data input to maximize branch coverage for focal method " + fm_name + ". " + "".join(
        ["/*FC*/", fc_name, "\n{", "/*FM*/ ", executed_fm, "/*F*/", executed_f, "/*C*/",
         constructors, "/*STUB*/", "/*M*/", methods, "\n}"
         ])
    return prompt


model = None


@app.on_event("startup")
async def load_model():
    global model
    model = load_model_by_type(model_name)


@app.post("/aka/generate")
async def predict(request: TestDataRequestModel, version_id: Optional[str] = Query(None)):
    if not request:
        raise HTTPException(status_code=400, detail="Input không được để trống")
    prompt = make_prompt(request)
    result = model.generate_from_prompt(prompt)
    return JSONResponse(content={"testData": result, "prompt": prompt})


@app.get("/model_serving")
async def index():
    return JSONResponse(content={"success": True})


@app.get("/aka/default_version")
async def default_version():
    return JSONResponse(
        content={"model": {"name": model_name + version_id, "provider": "provider", "description": "description",
                           "contextLength": str(max_token_length)},
                 "parameters": {"apiKey": "apikey", "maxTokens": str(max_token_length)}})


@app.get("/aka/all_versions")
async def all_versions():
    return JSONResponse(
        content=[{"model": {"name": model_name + version_id, "provider": "provider", "description": "description",
                            "contextLength": str(max_token_length)},
                  "parameters": {"apiKey": "apikey", "maxTokens": str(max_token_length)}}])


if __name__ == "__main__":
    # Chạy server trên host 0.0.0.0 và port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
