"""
Qwen3-VL-Embedding Server with OpenAI-compatible API
Supports text, image, and multimodal embeddings for LiteLLM gateway.

Based on: https://github.com/QwenLM/Qwen3-VL-Embedding
"""

import os
import sys
import base64
import logging
import tempfile
from typing import Optional, List, Dict, Any, Union
from io import BytesIO

import torch
import numpy as np
from PIL import Image
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add the Qwen3-VL-Embedding source to path
sys.path.insert(0, "/app/qwen3vl")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-Embedding-2B")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "8192"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Initialize FastAPI app
app = FastAPI(
    title="Qwen3-VL-Embedding Server",
    description="OpenAI-compatible API for multimodal embeddings",
    version="1.0.0"
)

# Global model reference
embedding_model = None


def get_model():
    """Load or return cached embedding model."""
    global embedding_model
    if embedding_model is None:
        logger.info(f"Loading Qwen3-VL-Embedding model: {MODEL_NAME}")
        
        # Import from the official Qwen3-VL-Embedding repo
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
        
        # Use float16 for GTX 1070 Ti (Pascal doesn't support bfloat16)
        # Don't use flash_attention_2 as it requires Ampere or newer
        embedding_model = Qwen3VLEmbedder(
            model_name_or_path=MODEL_NAME,
            max_length=MAX_LENGTH,
            torch_dtype=torch.float16,
        )
        logger.info(f"Qwen3-VL-Embedding model loaded successfully on {DEVICE}")
    return embedding_model


def load_image_from_input(image_input: str) -> Image.Image:
    """Load image from URL, path, or base64."""
    if image_input.startswith("http://") or image_input.startswith("https://"):
        response = requests.get(image_input, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    elif image_input.startswith("data:image"):
        # Base64 encoded image
        header, base64_data = image_input.split(",", 1)
        image_data = base64.b64decode(base64_data)
        return Image.open(BytesIO(image_data))
    else:
        # Assume it's a file path
        return Image.open(image_input)


class EmbeddingInput(BaseModel):
    """OpenAI-compatible embedding input."""
    input: Union[str, List[str], List[Dict[str, Any]]]
    model: str = "Qwen/Qwen3-VL-Embedding-2B"
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    # Extended fields for multimodal (Qwen3-VL specific)
    instruction: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    return {
        "status": "healthy", 
        "model": MODEL_NAME, 
        "device": DEVICE,
        "gpu": gpu_name,
        "gpu_available": gpu_available
    }


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "qwen"
            },
            {
                "id": "Qwen/Qwen3-VL-Embedding-2B",
                "object": "model",
                "created": 1700000000,
                "owned_by": "qwen"
            },
            {
                "id": "text-embedding-3-small",  # Alias for compatibility
                "object": "model",
                "created": 1700000000,
                "owned_by": "qwen"
            }
        ]
    }


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def create_embeddings(request: EmbeddingInput):
    """
    Create embeddings (OpenAI-compatible endpoint).
    
    Supports:
    - Text input: string or list of strings
    - Multimodal input: list of dicts with 'text', 'image', 'video' keys
    
    Qwen3-VL-Embedding format:
    - {"text": "...", "instruction": "..."} - text with optional instruction
    - {"image": "url_or_path"} - image embedding
    - {"text": "...", "image": "..."} - multimodal embedding
    - {"video": "url_or_path"} - video embedding
    """
    try:
        model = get_model()
        
        # Parse input into Qwen3-VL format
        # Format: List[Dict] with keys: text, image, video, instruction, fps, max_frames
        inputs = []
        
        if isinstance(request.input, str):
            # Single text input
            input_item = {"text": request.input}
            if request.instruction:
                input_item["instruction"] = request.instruction
            inputs.append(input_item)
            
        elif isinstance(request.input, list):
            for item in request.input:
                if isinstance(item, str):
                    # Text string
                    input_item = {"text": item}
                    if request.instruction:
                        input_item["instruction"] = request.instruction
                    inputs.append(input_item)
                elif isinstance(item, dict):
                    # Already in Qwen3-VL multimodal format
                    input_item = dict(item)
                    # Add instruction if provided and not already present
                    if request.instruction and "instruction" not in input_item:
                        input_item["instruction"] = request.instruction
                    inputs.append(input_item)
        
        logger.info(f"Processing {len(inputs)} input(s)")
        
        # Generate embeddings using Qwen3VLEmbedder.process()
        # Returns: torch.Tensor of shape [batch_size, hidden_dim]
        embeddings = model.process(inputs, normalize=True)
        
        # Convert to numpy for serialization
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().float().numpy()
        
        # Apply dimension reduction if requested
        if request.dimensions and request.dimensions < embeddings.shape[1]:
            embeddings = embeddings[:, :request.dimensions]
            # Re-normalize after truncation
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
        
        # Format response (OpenAI-compatible)
        data = []
        for i, emb in enumerate(embeddings):
            emb_list = emb.tolist()
            
            if request.encoding_format == "base64":
                emb_bytes = np.array(emb_list, dtype=np.float32).tobytes()
                emb_encoded = base64.b64encode(emb_bytes).decode("utf-8")
                data.append({
                    "object": "embedding",
                    "embedding": emb_encoded,
                    "index": i
                })
            else:
                data.append({
                    "object": "embedding",
                    "embedding": emb_list,
                    "index": i
                })
        
        # Calculate approximate token usage
        total_tokens = sum(
            len(str(inp.get("text", "")).split()) + 
            (256 if "image" in inp else 0) +  # Approximate tokens for image
            (512 if "video" in inp else 0)    # Approximate tokens for video
            for inp in inputs
        )
        
        return JSONResponse({
            "object": "list",
            "data": data,
            "model": MODEL_NAME,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        })
    
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings/multimodal")
async def create_multimodal_embeddings(
    text: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    instruction: Optional[str] = Form(default=None),
):
    """
    Create embeddings from multimodal input (file upload support).
    This is a convenience endpoint for direct file uploads.
    """
    try:
        model = get_model()
        
        input_item = {}
        if text:
            input_item["text"] = text
        if instruction:
            input_item["instruction"] = instruction
        
        tmp_path = None
        if image:
            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                content = await image.read()
                tmp.write(content)
                tmp_path = tmp.name
                input_item["image"] = tmp_path
        
        if not input_item:
            raise HTTPException(status_code=400, detail="No input provided")
        
        # Generate embedding
        embeddings = model.process([input_item], normalize=True)
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().float().numpy()
        
        # Cleanup temp file
        if tmp_path:
            os.unlink(tmp_path)
        
        return JSONResponse({
            "object": "list",
            "data": [{
                "object": "embedding",
                "embedding": embeddings[0].tolist(),
                "index": 0
            }],
            "model": MODEL_NAME,
            "usage": {
                "prompt_tokens": 1,
                "total_tokens": 1
            }
        })
    
    except Exception as e:
        logger.error(f"Multimodal embedding error: {str(e)}", exc_info=True)
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup."""
    logger.info("=" * 60)
    logger.info("Qwen3-VL-Embedding Server starting...")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Max Length: {MAX_LENGTH}")
    logger.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info("=" * 60)
    
    try:
        get_model()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
