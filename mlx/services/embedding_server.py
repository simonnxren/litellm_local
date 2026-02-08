"""
MLX Embedding Server — OpenAI-compatible /v1/embeddings endpoint.

Loads embedding models via mlx_lm. Supports Qwen3-VL-Embedding (and any
other causal-LM converted to MLX) by extracting last-token hidden states
from the transformer backbone and L2-normalising them.

For dedicated encoder-style models (BERT / BGE), falls back to
mlx_embeddings if available.

The Qwen3-VL-Embedding family uses a chat-template prompt format:
    system: <instruction>
    user:   <text to embed>
with `add_generation_prompt=True`.  The last hidden state of the final
token is the embedding vector (dim = hidden_size, e.g. 4096 for 8B).

Usage:
    python embedding_server.py                               # defaults
    python embedding_server.py --model <hf_repo> --port 8100
    MLX_EMBEDDING_MODEL=<repo> python embedding_server.py

Environment variables:
    MLX_EMBEDDING_MODEL   HuggingFace repo or local path
    MLX_EMBEDDING_PORT    Listen port   (default: 8100)
    MLX_EMBEDDING_HOST    Bind address  (default: 0.0.0.0)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, List, Optional, Union

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("mlx-embedding-server")

# ---------------------------------------------------------------------------
# Globals (populated at startup)
# ---------------------------------------------------------------------------
_model: Any = None
_tokenizer: Any = None
_model_name: str = ""
_embedding_dim: int = 0
_default_instruction: str = "Represent the user's input."


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _get_hidden_state_extractor(model):
    """Walk the model tree to find the transformer backbone that returns
    hidden states (before the lm_head projection).

    Qwen3-VL-Embedding structure:
        model (Qwen3VL wrapper)
          └── language_model (causal-LM)
                ├── model (Qwen3Model)  ← returns (batch, seq, hidden_size)
                └── lm_head             ← projects to vocab logits

    We want `language_model.model`.  Other architectures may differ, so
    we probe several common patterns.
    """
    # Pattern 1: model.language_model.model  (Qwen3-VL)
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", None)
        if inner is not None:
            return inner

    # Pattern 2: model.model  (most LLama / Mistral style)
    inner = getattr(model, "model", None)
    if inner is not None and not callable(inner):
        return inner

    # Pattern 3: model.transformer  (GPT-style)
    inner = getattr(model, "transformer", None)
    if inner is not None:
        return inner

    return None


def _load_model(model_path: str):
    global _model, _tokenizer, _model_name, _embedding_dim

    _model_name = model_path

    from mlx_lm import load as lm_load

    logger.info("Loading model via mlx_lm: %s", model_path)
    _model, _tokenizer = lm_load(model_path)
    logger.info("Model loaded: %s", type(_model).__name__)

    # Probe hidden dimension via a forward pass
    backbone = _get_hidden_state_extractor(_model)
    if backbone is None:
        raise RuntimeError(
            f"Cannot find transformer backbone in {type(_model).__name__}. "
            "Embedding extraction is not supported for this architecture."
        )

    probe_tokens = _tokenizer.encode("hello")
    x = mx.array([probe_tokens])
    h = backbone(x)
    if isinstance(h, tuple):
        h = h[0]
    _embedding_dim = int(h.shape[-1])
    logger.info("Embedding dim: %d  (backbone: %s)", _embedding_dim, type(backbone).__name__)


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def _format_prompt(text: str, instruction: str) -> str:
    """Build a chat-template prompt for the embedding model.

    Qwen3-VL-Embedding expects:
        system: <instruction>
        user:   <text>
    with generation_prompt appended so the last token is the assistant turn marker.
    """
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": text},
    ]
    try:
        return _tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    except Exception:
        # Fallback: plain text (for models without a chat template)
        return text


def _embed_texts(
    texts: List[str],
    instruction: str = "",
    dimensions: Optional[int] = None,
) -> np.ndarray:
    """Return (N, dim) float32 embeddings via last-token hidden-state pooling."""
    instruction = instruction or _default_instruction
    backbone = _get_hidden_state_extractor(_model)
    results: list[np.ndarray] = []

    for text in texts:
        prompt = _format_prompt(text, instruction)
        tokens = _tokenizer.encode(prompt)
        x = mx.array([tokens])

        # Forward through backbone → (1, seq_len, hidden_size)
        h = backbone(x)
        if isinstance(h, tuple):
            h = h[0]

        # Last-token pooling  (standard for Qwen-Embedding)
        emb = h[0, -1, :]  # (hidden_size,)

        # L2 normalise
        norm = mx.sqrt(mx.sum(emb * emb))
        emb = emb / mx.maximum(norm, mx.array(1e-12))

        results.append(np.array(emb, dtype=np.float32))

    embeddings = np.stack(results)

    # Optional MRL dimension truncation + re-normalise
    if dimensions and 0 < dimensions < embeddings.shape[1]:
        embeddings = embeddings[:, :dimensions]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)

    return embeddings


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text(s) to embed")
    model: str = Field("", description="Model name (informational)")
    encoding_format: str = Field("float", description="float or base64")
    dimensions: Optional[int] = Field(None, description="Truncate to N dims (MRL)")
    instruction: Optional[str] = Field(None, description="Task instruction for the embedding model")


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class UsageInfo(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: UsageInfo


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "mlx"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelObject]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup."""
    model_path = os.getenv("MLX_EMBEDDING_MODEL", app.state.model_path)
    logger.info("Loading embedding model: %s", model_path)
    _load_model(model_path)
    logger.info("✅  Embedding server ready  (model=%s, dim=%d)",
                _model_name, _embedding_dim)
    yield


app = FastAPI(
    title="MLX Embedding Server",
    description="OpenAI-compatible /v1/embeddings on Apple Silicon",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": _model_name,
        "embedding_dim": _embedding_dim,
    }


@app.get("/v1/models", response_model=ModelsResponse)
@app.get("/models", response_model=ModelsResponse)
async def list_models():
    return ModelsResponse(
        data=[
            ModelObject(id=_model_name, created=int(time.time()), owned_by="mlx"),
        ]
    )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """OpenAI-compatible embedding endpoint."""
    try:
        # Normalise input to list
        texts = [request.input] if isinstance(request.input, str) else request.input
        if not texts:
            raise HTTPException(status_code=400, detail="Empty input")

        t0 = time.perf_counter()
        embeddings = _embed_texts(
            texts,
            instruction=request.instruction or "",
            dimensions=request.dimensions,
        )
        elapsed = time.perf_counter() - t0
        logger.info("Embedded %d text(s) in %.3fs  dim=%d", len(texts), elapsed, embeddings.shape[1])

        # Count tokens (rough estimate)
        total_tokens = sum(len(t.split()) for t in texts)

        data = [
            EmbeddingObject(
                embedding=emb.tolist(),
                index=i,
            )
            for i, emb in enumerate(embeddings)
        ]
        return EmbeddingResponse(
            data=data,
            model=_model_name,
            usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Embedding error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MLX Embedding Server")
    parser.add_argument(
        "--model",
        default=os.getenv("MLX_EMBEDDING_MODEL", "jedisct1/Qwen3-VL-Embedding-8B-mlx"),
        help="HuggingFace repo or local path (default: jedisct1/Qwen3-VL-Embedding-8B-mlx)",
    )
    parser.add_argument("--host", default=os.getenv("MLX_EMBEDDING_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MLX_EMBEDDING_PORT", "8100")))
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn workers (keep 1 for MLX)")
    args = parser.parse_args()

    # Stash model path for lifespan to pick up
    app.state.model_path = args.model
    os.environ.setdefault("MLX_EMBEDDING_MODEL", args.model)

    logger.info("Starting MLX Embedding Server on %s:%d  model=%s", args.host, args.port, args.model)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
