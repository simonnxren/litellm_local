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
# Audio chunking config for long audio with timestamps
CHUNK_DURATION_SECONDS = int(
    os.environ.get("CHUNK_DURATION_SECONDS", "60")
)  # 60 second chunks
CHUNK_OVERLAP_SECONDS = float(
    os.environ.get("CHUNK_OVERLAP_SECONDS", "1.0")
)  # 1 second overlap

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


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    info = sf.info(audio_path)
    return info.duration


def chunk_audio(
    audio_path: str, chunk_duration: float = 60.0, overlap: float = 1.0
) -> List[tuple]:
    """
    Split audio file into chunks for processing long audio with limited GPU memory.

    Args:
        audio_path: Path to audio file
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds (helps with word boundaries)

    Returns:
        List of (audio_array, sample_rate, start_time_offset) tuples
    """
    audio, sr = sf.read(audio_path, dtype="float32")

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    total_samples = len(audio)
    total_duration = total_samples / sr

    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    step_samples = chunk_samples - overlap_samples

    chunks = []
    start_sample = 0

    while start_sample < total_samples:
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk_audio = audio[start_sample:end_sample]
        start_time = start_sample / sr

        chunks.append((chunk_audio, sr, start_time))

        start_sample += step_samples

        # If remaining audio is very short, include it in last chunk
        if total_samples - start_sample < sr * 2:  # Less than 2 seconds remaining
            break

    print(
        f"Split {total_duration:.1f}s audio into {len(chunks)} chunks of ~{chunk_duration}s each"
    )
    return chunks


def transcribe_chunked(
    model,
    audio_path: str,
    language: Optional[str] = None,
    return_timestamps: bool = False,
    chunk_duration: float = 60.0,
) -> dict:
    """
    Transcribe long audio by processing in chunks to avoid GPU OOM.

    Args:
        model: Qwen3ASRModel instance
        audio_path: Path to audio file
        language: Optional language hint
        return_timestamps: Whether to return word/segment timestamps
        chunk_duration: Duration of each chunk in seconds

    Returns:
        Combined transcription result with merged timestamps
    """
    import gc

    chunks = chunk_audio(
        audio_path, chunk_duration=chunk_duration, overlap=CHUNK_OVERLAP_SECONDS
    )

    all_text_parts = []
    all_words = []
    detected_language = None

    for i, (chunk_data, sr, time_offset) in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)} (offset: {time_offset:.1f}s)...")

        # Clear GPU cache before each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        try:
            results = model.transcribe(
                audio=(chunk_data, sr),
                language=language,
                return_time_stamps=return_timestamps,
            )

            result = results[0]
            all_text_parts.append(result.text)

            if detected_language is None:
                detected_language = result.language

            # Adjust timestamps by chunk offset
            if result.time_stamps:
                for ts in result.time_stamps:
                    all_words.append(
                        {
                            "word": ts.text,
                            "start": ts.start_time + time_offset,
                            "end": ts.end_time + time_offset,
                        }
                    )

        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")
            # Continue with other chunks even if one fails
            all_text_parts.append("")

    # Combine results
    full_text = " ".join(all_text_parts).strip()
    # Clean up double spaces
    while "  " in full_text:
        full_text = full_text.replace("  ", " ")

    return {
        "text": full_text,
        "language": detected_language or "unknown",
        "words": all_words if return_timestamps else None,
    }


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


def words_to_segments(
    words: List[dict], max_gap: float = 0.5, max_segment_duration: float = 30.0
) -> List[dict]:
    """
    Aggregate word-level timestamps into segment-level timestamps.

    Segments are created based on:
    1. Sentence-ending punctuation (. ! ? 。 ！ ？ etc.)
    2. Silence gaps between words (> max_gap seconds)
    3. Maximum segment duration (to avoid very long segments)

    This mimics Whisper's segment output format for OpenAI API compatibility.

    Args:
        words: List of word dicts with 'word', 'start', 'end' keys
        max_gap: Maximum gap (seconds) between words to stay in same segment (default: 0.5s)
        max_segment_duration: Maximum segment duration before forcing a split (default: 30s)

    Returns:
        List of segment dicts with 'id', 'start', 'end', 'text' keys (OpenAI format)
    """
    if not words:
        return []

    segments = []
    current_segment_words = []
    current_start = None
    current_end = None
    segment_id = 0

    # Sentence-ending punctuation (including Chinese, Japanese, etc.)
    sentence_endings = {".", "!", "?", "。", "！", "？", "…", "；", ";"}

    for word_info in words:
        word = word_info.get("word", "")
        start = word_info.get("start", 0)
        end = word_info.get("end", 0)

        # Check if previous word ended with sentence-ending punctuation
        prev_ends_sentence = (
            current_segment_words
            and current_segment_words[-1].rstrip()[-1:] in sentence_endings
        )

        # Start a new segment if:
        # 1. This is the first word
        # 2. Gap between words is too large (silence/pause)
        # 3. Current segment is too long
        # 4. Previous word ended with sentence-ending punctuation
        should_start_new = (
            current_start is None
            or (current_end is not None and (start - current_end) > max_gap)
            or (
                current_start is not None
                and (end - current_start) > max_segment_duration
            )
            or prev_ends_sentence
        )

        if should_start_new and current_segment_words:
            # Save current segment - join with spaces for readability
            segment_text = " ".join(current_segment_words).strip()
            # Clean up spacing around punctuation
            for punct in ".,!?;:。，！？；：":
                segment_text = segment_text.replace(f" {punct}", punct)
            if segment_text:
                segments.append(
                    {
                        "id": segment_id,
                        "start": current_start,
                        "end": current_end,
                        "text": segment_text,
                    }
                )
                segment_id += 1
            current_segment_words = []
            current_start = None

        # Add word to current segment
        current_segment_words.append(word)
        if current_start is None:
            current_start = start
        current_end = end

    # Don't forget the last segment
    if current_segment_words:
        segment_text = " ".join(current_segment_words).strip()
        # Clean up spacing around punctuation
        for punct in ".,!?;:。，！？；：":
            segment_text = segment_text.replace(f" {punct}", punct)
        if segment_text:
            segments.append(
                {
                    "id": segment_id,
                    "start": current_start,
                    "end": current_end,
                    "text": segment_text,
                }
            )

    return segments


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

    Supports timestamp_granularities: ["word"], ["segment"], or ["word", "segment"]
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
            if not timestamp_granularities:
                # Try without brackets
                tg = request.form.get("timestamp_granularities")
                if tg:
                    # Could be JSON array or single value
                    try:
                        timestamp_granularities = json.loads(tg)
                    except json.JSONDecodeError:
                        timestamp_granularities = [tg]

            return_timestamps = (
                len(timestamp_granularities) > 0
                or request.form.get("return_timestamps", "").lower() == "true"
            )

            want_words = "word" in timestamp_granularities
            want_segments = "segment" in timestamp_granularities

            # Read audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_file.save(tmp.name)
                audio_path = tmp.name

            try:
                # Check if we need chunked processing
                # Use chunking for long audio with timestamps to avoid GPU OOM
                audio_duration = get_audio_duration(audio_path)
                use_chunking = (
                    return_timestamps and audio_duration > CHUNK_DURATION_SECONDS
                )

                if use_chunking:
                    print(
                        f"Using chunked processing for {audio_duration:.1f}s audio with timestamps"
                    )
                    chunked_result = transcribe_chunked(
                        asr_model,
                        audio_path,
                        language=language,
                        return_timestamps=True,
                        chunk_duration=CHUNK_DURATION_SECONDS,
                    )

                    # Create a mock result object for unified handling
                    class MockResult:
                        def __init__(self, text, language, words):
                            self.text = text
                            self.language = language
                            self.time_stamps = None
                            self._words = words

                    result = MockResult(
                        chunked_result["text"],
                        chunked_result["language"],
                        chunked_result.get("words", []),
                    )
                    results = [result]
                    # Store words for later processing
                    chunked_words = chunked_result.get("words", [])
                else:
                    results = asr_model.transcribe(
                        audio=audio_path,
                        language=language,
                        return_time_stamps=return_timestamps,
                    )
                    chunked_words = None
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
            context = data.get("context", "")

            # Parse timestamp_granularities
            timestamp_granularities = data.get("timestamp_granularities", [])
            if isinstance(timestamp_granularities, str):
                timestamp_granularities = [timestamp_granularities]

            return_timestamps = len(timestamp_granularities) > 0 or data.get(
                "return_timestamps", False
            )

            want_words = "word" in timestamp_granularities
            want_segments = "segment" in timestamp_granularities

            results = asr_model.transcribe(
                audio=audio,
                language=language,
                context=context,
                return_time_stamps=return_timestamps,
            )
            chunked_words = None  # JSON path doesn't support chunking yet

        # Format response
        result = results[0]
        response: dict[str, Any] = {
            "text": result.text,
            "language": result.language,
        }

        # Handle timestamps - either from chunked processing or regular transcription
        if chunked_words is not None:
            # Chunked processing - words already in dict format
            words = chunked_words

            if want_words or (not want_words and not want_segments):
                response["words"] = words

            if want_segments:
                response["segments"] = words_to_segments(words)

        elif result.time_stamps:
            # Regular processing - convert from timestamp objects
            words = [
                {
                    "word": ts.text,
                    "start": ts.start_time,
                    "end": ts.end_time,
                }
                for ts in result.time_stamps
            ]

            # Include words if requested or if no specific granularity was requested
            if want_words or (not want_words and not want_segments):
                response["words"] = words

            # Include segments if requested
            if want_segments:
                response["segments"] = words_to_segments(words)

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
