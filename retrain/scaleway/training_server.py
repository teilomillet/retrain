"""FastAPI training server — runs on the Scaleway GPU instance.

Exposes LocalTrainHelper over HTTP so retrain can drive training remotely.

Usage (on the GPU instance):
    retrain-training-server --port 8001 --model meta-llama/Llama-3.1-8B-Instruct --lora-rank 32
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="retrain-training-server")

# Populated by main() before uvicorn starts
_helper = None
_inference_url: str = "http://localhost:8000"
_inference_engine: str = "vllm"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class TrainStepRequest(BaseModel):
    tokens: list[list[int]]
    logprobs: list[list[float]]
    advantages: list[list[float]]
    lr: float
    weight_decay: float = 0.01


class CheckpointRequest(BaseModel):
    name: str


class SaveAdapterRequest(BaseModel):
    path: str
    name: str


class LoadStateRequest(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train_step")
def train_step(req: TrainStepRequest) -> dict[str, float]:
    if _helper is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="helper not initialized")
    loss = _helper.train_step(
        req.tokens,
        req.logprobs,
        req.advantages,
        req.lr,
        req.weight_decay,
    )
    return {"loss": float(loss)}


@app.post("/checkpoint")
def checkpoint(req: CheckpointRequest) -> dict:
    if _helper is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="helper not initialized")
    _helper.checkpoint(req.name)
    _reload_lora_on_inference(req.name)
    return {}


@app.post("/save_adapter")
def save_adapter(req: SaveAdapterRequest) -> Response:
    if _helper is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="helper not initialized")
    saved_path = _helper.save_adapter(req.path, req.name)
    # Read all safetensors files and pack them as a tar archive
    import io
    import tarfile
    buf = io.BytesIO()
    adapter_dir = Path(saved_path)
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in adapter_dir.rglob("*"):
            if f.is_file():
                tar.add(f, arcname=f.relative_to(adapter_dir))
    return Response(content=buf.getvalue(), media_type="application/octet-stream")


@app.post("/load_state")
def load_state(req: LoadStateRequest) -> dict:
    if _helper is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="helper not initialized")
    _helper.load_state(req.name)
    return {}


# ---------------------------------------------------------------------------
# LoRA reload on inference engine
# ---------------------------------------------------------------------------

def _reload_lora_on_inference(name: str) -> None:
    # The adapter was just saved by checkpoint(); we need to tell the inference
    # engine to reload it. vLLM and SGLang use different endpoints.
    try:
        if _inference_engine == "vllm":
            httpx.post(
                f"{_inference_url}/v1/load_lora_adapter",
                json={"lora_name": name, "lora_path": name},
                timeout=30,
            ).raise_for_status()
        else:
            httpx.post(
                f"{_inference_url}/add_lora",
                json={"lora_name": name, "lora_path": name},
                timeout=30,
            ).raise_for_status()
    except Exception as exc:
        logger.warning("LoRA reload on inference engine failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="retrain training server")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--adapter-path", default="/tmp/retrain_adapter")
    parser.add_argument("--inference-url", default="http://localhost:8000")
    parser.add_argument("--inference-engine", default="vllm", choices=["vllm", "sglang"])
    args = parser.parse_args()

    global _helper, _inference_url, _inference_engine
    _inference_url = args.inference_url
    _inference_engine = args.inference_engine

    from retrain.local_train_helper import LocalTrainHelper
    _helper = LocalTrainHelper(
        args.model,
        args.adapter_path,
        "gpu:0",
        args.lora_rank,
        engine_type="pytorch",
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
