#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-ASR Transformers Inference Server

A standalone script to serve Qwen3-ASR with Transformers backend, providing both
OpenAI-compatible API and direct transcription endpoints.

Usage:
    # Development mode (Flask dev server)
    python serve.py

    # Production mode (Gunicorn)
    gunicorn -w 1 -b 0.0.0.0:8000 --timeout 300 'serve:create_app()'

Environment Variables:
    ASR_MODEL: Model path (default: Qwen/Qwen3-ASR-1.7B)
    ALIGNER_MODEL: Forced aligner model path (default: Qwen/Qwen3-ForcedAligner-0.6B)
    MAX_BATCH_SIZE: Max inference batch size (default: 32)
    MAX_NEW_TOKENS: Max tokens to generate (default: 256)
    SERVER_HOST: Server host (default: 0.0.0.0)
    SERVER_PORT: Server port (default: 8000)
    DEVICE_MAP: Device mapping (default: cuda:0)
    DTYPE: Data type - bfloat16, float16, float32 (default: bfloat16)
    GUNICORN_WORKERS: Number of gunicorn workers (default: 1, keep at 1 for GPU)
    GUNICORN_TIMEOUT: Request timeout in seconds (default: 300)
"""

import os
import sys
import json
import base64
import tempfile
from typing import Optional, List, Union, Any
from io import BytesIO

import numpy as np
import soundfile as sf
import torch
from flask import Flask, request, jsonify

# Configuration from environment
ASR_MODEL = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
ALIGNER_MODEL = os.environ.get("ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))
DEVICE_MAP = os.environ.get("DEVICE_MAP", "cuda:0")
DTYPE = os.environ.get("DTYPE", "bfloat16")

# Map string dtype to torch dtype
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Initialize Flask app
app = Flask(__name__)

# Global model instance
model = None


def get_model():
    """Lazy load the model using Transformers backend."""
    global model
    if model is None:
        from qwen_asr import Qwen3ASRModel

        dtype = DTYPE_MAP.get(DTYPE, torch.bfloat16)

        print(f"Loading ASR model: {ASR_MODEL}")
        print(f"Loading Aligner model: {ALIGNER_MODEL}")
        print(f"Device map: {DEVICE_MAP}")
        print(f"Dtype: {DTYPE}")

        model = Qwen3ASRModel.from_pretrained(
            ASR_MODEL,
            dtype=dtype,
            device_map=DEVICE_MAP,
            max_inference_batch_size=MAX_BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            forced_aligner=ALIGNER_MODEL,
            forced_aligner_kwargs=dict(
                dtype=dtype,
                device_map=DEVICE_MAP,
            ),
        )
        print("Models loaded successfully!")
    return model


def decode_audio_input(audio_input: Union[str, dict]) -> Union[str, tuple]:
    """
    Decode various audio input formats.

    Supports:
    - URL string
    - Base64 data URL
    - File path
    - Dict with 'url' or 'data' key
    """
    if isinstance(audio_input, dict):
        if "url" in audio_input:
            audio_input = audio_input["url"]
        elif "data" in audio_input:
            audio_input = audio_input["data"]

    if not isinstance(audio_input, str):
        return audio_input  # type: ignore

    # Check if it's a base64 data URL
    if audio_input.startswith("data:"):
        # Parse data URL: data:audio/wav;base64,xxxxx
        try:
            header, data = audio_input.split(",", 1)
            audio_bytes = base64.b64decode(data)
            wav, sr = sf.read(BytesIO(audio_bytes), dtype="float32")
            return (np.asarray(wav, dtype=np.float32), int(sr))
        except Exception as e:
            raise ValueError(f"Failed to decode base64 audio: {e}")

    # Return as-is (URL or file path)
    return audio_input


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": ASR_MODEL, "backend": "transformers"})


@app.route("/v1/models", methods=["GET"])
def list_models():
    """OpenAI-compatible models endpoint."""
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": ASR_MODEL,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "qwen",
                    "permission": [],
                    "root": ASR_MODEL,
                    "parent": None,
                }
            ],
        }
    )


@app.route("/v1/audio/transcriptions", methods=["POST"])
def transcribe():
    """
    OpenAI-compatible transcription endpoint.

    Accepts:
    - multipart/form-data with 'file' field
    - application/json with 'audio' field (URL or base64)
    """
    try:
        asr_model = get_model()

        # Handle multipart form data (file upload)
        if request.content_type and "multipart/form-data" in request.content_type:
            if "file" not in request.files:
                return jsonify({"error": "No file provided"}), 400

            audio_file = request.files["file"]
            language = request.form.get("language")
            # Check for OpenAI-style timestamp_granularities[] or timestamp_granularities
            # Flask parses timestamp_granularities[] as multiple values with getlist()
            timestamp_granularities = request.form.getlist("timestamp_granularities[]")
            return_timestamps = (
                len(timestamp_granularities) > 0
                or request.form.get("timestamp_granularities") is not None
                or request.form.get("return_timestamps", "").lower() == "true"
            )

            # Read audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_file.save(tmp.name)
                audio_path = tmp.name

            try:
                results = asr_model.transcribe(
                    audio=audio_path,
                    language=language,
                    return_time_stamps=return_timestamps,
                )
            finally:
                os.unlink(audio_path)

        # Handle JSON request
        else:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            audio_input = data.get("audio") or data.get("audio_url") or data.get("url")
            if not audio_input:
                return jsonify({"error": "No audio input provided"}), 400

            audio = decode_audio_input(audio_input)
            language = data.get("language")
            return_timestamps = data.get("return_timestamps", False)
            context = data.get("context", "")

            results = asr_model.transcribe(
                audio=audio,
                language=language,
                context=context,
                return_time_stamps=return_timestamps,
            )

        # Format response
        result = results[0]
        response: dict[str, Any] = {
            "text": result.text,
            "language": result.language,
        }

        if result.time_stamps:
            response["words"] = [
                {
                    "word": ts.text,
                    "start": ts.start_time,
                    "end": ts.end_time,
                }
                for ts in result.time_stamps
            ]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe", methods=["POST"])
def transcribe_batch():
    """
    Batch transcription endpoint.

    Request body:
    {
        "audios": ["url1", "url2", ...],
        "languages": ["English", null, ...],  # optional
        "contexts": ["", "", ...],  # optional
        "return_timestamps": false
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        audios = data.get("audios", [])
        if not audios:
            return jsonify({"error": "No audios provided"}), 400

        # Decode all audio inputs
        decoded_audios = [decode_audio_input(a) for a in audios]

        languages = data.get("languages")
        contexts = data.get("contexts")
        return_timestamps = data.get("return_timestamps", False)

        asr_model = get_model()

        results = asr_model.transcribe(
            audio=decoded_audios,
            language=languages,
            context=contexts,
            return_time_stamps=return_timestamps,
        )

        # Format response
        response_results = []
        for result in results:
            item: dict[str, Any] = {
                "text": result.text,
                "language": result.language,
            }
            if result.time_stamps:
                item["words"] = [
                    {
                        "word": ts.text,
                        "start": ts.start_time,
                        "end": ts.end_time,
                    }
                    for ts in result.time_stamps
                ]
            response_results.append(item)

        return jsonify({"results": response_results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    """API documentation."""
    return jsonify(
        {
            "name": "Qwen3-ASR Transformers Server",
            "model": ASR_MODEL,
            "aligner": ALIGNER_MODEL,
            "backend": "transformers",
            "device": DEVICE_MAP,
            "dtype": DTYPE,
            "endpoints": {
                "/health": "GET - Health check",
                "/v1/audio/transcriptions": "POST - OpenAI-compatible transcription",
                "/transcribe": "POST - Batch transcription",
            },
            "examples": {
                "transcribe_url": {
                    "method": "POST",
                    "url": "/v1/audio/transcriptions",
                    "body": {
                        "audio": "https://example.com/audio.wav",
                        "language": "English",
                        "return_timestamps": True,
                    },
                },
                "transcribe_batch": {
                    "method": "POST",
                    "url": "/transcribe",
                    "body": {
                        "audios": ["url1", "url2"],
                        "languages": ["English", "Chinese"],
                        "return_timestamps": False,
                    },
                },
            },
        }
    )


def create_app():
    """
    Application factory for Gunicorn/WSGI servers.

    Pre-loads the model during app creation to ensure it's ready
    before the first request.

    Usage with gunicorn:
        gunicorn -w 1 -b 0.0.0.0:8000 --timeout 300 'serve:create_app()'
    """
    print(f"Creating Qwen3-ASR app...")
    print(f"ASR Model: {ASR_MODEL}")
    print(f"Aligner Model: {ALIGNER_MODEL}")
    print(f"Device: {DEVICE_MAP}")
    print(f"Dtype: {DTYPE}")

    # Pre-load model during app creation
    get_model()

    return app


if __name__ == "__main__":
    print(f"Starting Qwen3-ASR Transformers Server on {SERVER_HOST}:{SERVER_PORT}")
    print(f"ASR Model: {ASR_MODEL}")
    print(f"Aligner Model: {ALIGNER_MODEL}")
    print(f"Device: {DEVICE_MAP}")
    print(f"Dtype: {DTYPE}")
    print("NOTE: For production, use gunicorn instead:")
    print(
        f"  gunicorn -w 1 -b {SERVER_HOST}:{SERVER_PORT} --timeout 300 'serve:create_app()'"
    )
    print("")

    # Pre-load model
    get_model()

    # Run development server
    app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True)
