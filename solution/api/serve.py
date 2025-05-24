#!/usr/bin/env python3
"""
API server for SLM summarization using the ONNX model.
"""
import os
import argparse
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Union
from src.onnx_export import generate_summary

# Simple in-memory cache
cache = {}

# Initialize FastAPI app
app = FastAPI(title="SLM ONNX Summarization API")

# Mount the UI directory to serve index.html and assets
ui_dir = os.path.join(os.path.dirname(__file__), "ui")
if os.path.isdir(ui_dir):
    app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

class SummarizeRequest(BaseModel):
    texts: Union[str, List[str]]

class SummarizeResponse(BaseModel):
    summaries: List[str]

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    # Accept single string or list of strings
    inputs = request.texts if isinstance(request.texts, list) else [request.texts]
    outputs = []
    model_dir = os.getenv("MODEL_DIR", "models/onnx_model")
    model_type = os.getenv("MODEL_TYPE", "base")
    use_gpu = os.getenv("USE_GPU", "false").lower() in ("1", "true", "yes")

    for text in inputs:
        if not text:
            outputs.append("")
            continue
        if text in cache:
            outputs.append(cache[text])
            continue
        try:
            summary = generate_summary(
                text,
                model_dir=model_dir,
                model_type=model_type,
                use_gpu=use_gpu
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        cache[text] = summary
        outputs.append(summary)

    return SummarizeResponse(summaries=outputs)

def main():
    parser = argparse.ArgumentParser(description="Serve SLM summarization API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
