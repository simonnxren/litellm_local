#!/usr/bin/env python3
"""
Qwen3-ASR Server - Simple transcription with timestamps

Endpoints:
    GET  /health                    - Health check
    GET  /v1/models                 - List models (OpenAI-compatible)
    POST /v1/audio/transcriptions   - Transcribe audio with timestamps
"""

import os
import base64
import tempfile
from io import BytesIO
from typing import Optional, Any

import numpy as np
import soundfile as sf
import torch
from flask import Flask, request, jsonify

# Configuration
ASR_MODEL = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
ALIGNER_MODEL = os.environ.get("ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
DEVICE_MAP = os.environ.get("DEVICE_MAP", "cuda:0")
DTYPE = os.environ.get("DTYPE", "bfloat16")
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

app = Flask(__name__)
model = None


def get_model():
    """Lazy load the ASR model with ForcedAligner for timestamps."""
    global model
    if model is None:
        from qwen_asr import Qwen3ASRModel
        dtype = DTYPE_MAP.get(DTYPE, torch.bfloat16)
        print(f"Loading ASR: {ASR_MODEL}, Aligner: {ALIGNER_MODEL}, dtype: {dtype}")
        model = Qwen3ASRModel.from_pretrained(
            ASR_MODEL,
            dtype=dtype,
            device_map=DEVICE_MAP,
            forced_aligner=ALIGNER_MODEL,
            forced_aligner_kwargs=dict(dtype=dtype, device_map=DEVICE_MAP),
        )
        print("Models loaded!")
    return model


def decode_audio(audio_input) -> tuple:
    """Decode audio from base64 data URL or return path/URL."""
    if isinstance(audio_input, dict):
        audio_input = audio_input.get("url") or audio_input.get("data")
    
    if isinstance(audio_input, str) and audio_input.startswith("data:"):
        _, data = audio_input.split(",", 1)
        audio_bytes = base64.b64decode(data)
        wav, sr = sf.read(BytesIO(audio_bytes), dtype="float32")
        return (np.asarray(wav, dtype=np.float32), int(sr))
    
    return audio_input


def words_to_segments(words: list, max_gap: float = 0.5) -> list:
    """Convert word timestamps to sentence segments."""
    if not words:
        return []
    
    segments = []
    current_words = []
    current_start = None
    current_end = None
    seg_id = 0
    endings = {".", "!", "?", "。", "！", "？"}
    
    for w in words:
        word, start, end = w["word"], w["start"], w["end"]
        prev_ends = current_words and current_words[-1].rstrip()[-1:] in endings
        
        if current_start is None or prev_ends or (current_end and start - current_end > max_gap):
            if current_words:
                text = " ".join(current_words).strip()
                for p in ".,!?。，！？":
                    text = text.replace(f" {p}", p)
                segments.append({"id": seg_id, "start": current_start, "end": current_end, "text": text})
                seg_id += 1
            current_words = []
            current_start = None
        
        current_words.append(word)
        if current_start is None:
            current_start = start
        current_end = end
    
    if current_words:
        text = " ".join(current_words).strip()
        for p in ".,!?。，！？":
            text = text.replace(f" {p}", p)
        segments.append({"id": seg_id, "start": current_start, "end": current_end, "text": text})
    
    return segments


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": ASR_MODEL})


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{"id": ASR_MODEL, "object": "model", "owned_by": "qwen"}]
    })


@app.route("/v1/audio/transcriptions", methods=["POST"])
def transcribe():
    """
    OpenAI-compatible transcription endpoint.
    
    Accepts multipart/form-data with 'file' or JSON with 'audio'.
    Supports timestamp_granularities: ["word"], ["segment"], or both.
    """
    try:
        asr = get_model()
        
        # Parse request
        if request.content_type and "multipart/form-data" in request.content_type:
            if "file" not in request.files:
                return jsonify({"error": "No file provided"}), 400
            
            audio_file = request.files["file"]
            language = request.form.get("language")
            
            granularities = request.form.getlist("timestamp_granularities[]")
            if not granularities:
                tg = request.form.get("timestamp_granularities")
                if tg:
                    import json
                    try:
                        granularities = json.loads(tg)
                    except:
                        granularities = [tg]
            
            return_ts = len(granularities) > 0 or request.form.get("return_timestamps", "").lower() == "true"
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_file.save(tmp.name)
                audio_path = tmp.name
            
            try:
                results = asr.transcribe(audio=audio_path, language=language, return_time_stamps=return_ts)
            finally:
                os.unlink(audio_path)
        else:
            data = request.get_json() or {}
            audio_input = data.get("audio") or data.get("audio_url") or data.get("url")
            if not audio_input:
                return jsonify({"error": "No audio provided"}), 400
            
            audio = decode_audio(audio_input)
            language = data.get("language")
            granularities = data.get("timestamp_granularities", [])
            if isinstance(granularities, str):
                granularities = [granularities]
            return_ts = len(granularities) > 0 or data.get("return_timestamps", False)
            
            results = asr.transcribe(audio=audio, language=language, return_time_stamps=return_ts)
        
        # Format response
        result = results[0]
        response: dict[str, Any] = {"text": result.text, "language": result.language}
        
        want_words = "word" in granularities
        want_segments = "segment" in granularities
        
        if result.time_stamps:
            words = [{"word": ts.text, "start": ts.start_time, "end": ts.end_time} for ts in result.time_stamps]
            
            if want_words or (not want_words and not want_segments):
                response["words"] = words
            if want_segments:
                response["segments"] = words_to_segments(words)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "Qwen3-ASR Server",
        "model": ASR_MODEL,
        "aligner": ALIGNER_MODEL,
        "endpoints": ["/health", "/v1/models", "/v1/audio/transcriptions"]
    })


def create_app():
    """Factory for Gunicorn."""
    get_model()
    return app


if __name__ == "__main__":
    print(f"Starting on {SERVER_HOST}:{SERVER_PORT}")
    get_model()
    app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True)
