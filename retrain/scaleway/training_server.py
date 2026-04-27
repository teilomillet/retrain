"""FastAPI training server — runs on the Scaleway GPU instance.

Exposes LocalTrainHelper over HTTP so retrain can drive training remotely.

Usage (on the GPU instance):
    retrain-training-server --port 8001 --model meta-llama/Llama-3.1-8B-Instruct --lora-rank 32
"""

from __future__ import annotations

import argparse
import logging
import os
import tarfile
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server state (injected at startup via lifespan, accessed via request.app.state)
# ---------------------------------------------------------------------------

@dataclass
class ServerState:
    helper: object
    inference_url: str
    inference_engine: str
    adapter_base: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield


def create_app(state: ServerState) -> FastAPI:
    _app = FastAPI(title="retrain-training-server", lifespan=lifespan)
    _app.state.server = state
    return _app


# Module-level app instance used when launched via CLI (main())
app = FastAPI(title="retrain-training-server")


def _state(request: Request) -> ServerState:
    s = request.app.state.server
    if s is None or s.helper is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="helper not initialized")
    return s


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
    name: str


class LoadStateRequest(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health(request: Request) -> dict[str, str]:
    _state(request)
    return {"status": "ok"}


@app.post("/train_step")
def train_step(req: TrainStepRequest, request: Request) -> dict[str, float]:
    s = _state(request)
    loss = s.helper.train_step(req.tokens, req.logprobs, req.advantages, req.lr, req.weight_decay)
    return {"loss": float(loss)}


@app.post("/checkpoint")
def checkpoint(req: CheckpointRequest, request: Request) -> dict:
    s = _state(request)
    s.helper.checkpoint(req.name)
    _reload_lora_on_inference(s, req.name)
    return {}


@app.post("/save_adapter")
def save_adapter(req: SaveAdapterRequest, request: Request, background_tasks: BackgroundTasks) -> StreamingResponse:
    s = _state(request)
    saved_path = s.helper.save_adapter(s.adapter_base, req.name)
    adapter_dir = Path(saved_path)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
        with tarfile.open(fileobj=tmp, mode="w:gz") as tar:
            for f in adapter_dir.rglob("*"):
                if f.is_file():
                    tar.add(f, arcname=f.relative_to(adapter_dir))
    background_tasks.add_task(os.remove, tmp_path)

    def _iter_file(path: str, chunk_size: int = 1024 * 1024):
        with open(path, "rb") as fh:
            while chunk := fh.read(chunk_size):
                yield chunk

    return StreamingResponse(
        _iter_file(tmp_path),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{req.name}.tar.gz"'},
    )


@app.post("/load_state")
def load_state(req: LoadStateRequest, request: Request) -> dict:
    s = _state(request)
    s.helper.load_state(req.name)
    return {}


# ---------------------------------------------------------------------------
# LoRA reload on inference engine
# ---------------------------------------------------------------------------

def _reload_lora_on_inference(state: ServerState, name: str) -> None:
    lora_path = str(Path(state.adapter_base) / name)
    try:
        if state.inference_engine == "vllm":
            httpx.post(
                f"{state.inference_url}/v1/load_lora_adapter",
                json={"lora_name": name, "lora_path": lora_path},
                timeout=30,
            ).raise_for_status()
        else:
            httpx.post(
                f"{state.inference_url}/add_lora",
                json={"lora_name": name, "lora_path": lora_path},
                timeout=30,
            ).raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"LoRA reload on inference engine failed: {exc}") from exc


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

    from retrain.local_train_helper import LocalTrainHelper
    helper = LocalTrainHelper(
        args.model,
        args.adapter_path,
        "gpu:0",
        args.lora_rank,
        engine_type="pytorch",
    )

    state = ServerState(
        helper=helper,
        inference_url=args.inference_url,
        inference_engine=args.inference_engine,
        adapter_base=args.adapter_path,
    )
    server = create_app(state)
    uvicorn.run(server, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
