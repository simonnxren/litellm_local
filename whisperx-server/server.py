"""
WhisperX ASR Server with OpenAI-compatible API
Supports speaker diarization, word-level timestamps, and multiple output formats.
"""

import os
import io
import json
import tempfile
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

import torch

# Fix for PyTorch 2.6+ weights_only=True default
# Required for loading WhisperX/pyannote models that use omegaconf/typing
# Monkey-patch torch.load to use weights_only=False by default
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import whisperx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
ASR_MODEL = os.getenv("ASR_MODEL", "large-v3-turbo")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Initialize FastAPI app
app = FastAPI(
    title="WhisperX ASR Server",
    description="OpenAI-compatible API for speech-to-text with speaker diarization",
    version="1.0.0"
)

# Global model references
whisper_model = None
diarize_model = None
align_model = None
align_metadata = None


def get_whisper_model():
    """Load or return cached WhisperX model."""
    global whisper_model
    if whisper_model is None:
        logger.info(f"Loading WhisperX model: {ASR_MODEL} (compute_type={COMPUTE_TYPE}, device={DEVICE})")
        whisper_model = whisperx.load_model(
            ASR_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            language=None  # Auto-detect
        )
        logger.info("WhisperX model loaded successfully")
    return whisper_model


def get_diarize_model():
    """Load or return cached diarization model."""
    global diarize_model
    if diarize_model is None and HF_TOKEN:
        logger.info("Loading speaker diarization model...")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=HF_TOKEN,
            device=DEVICE
        )
        logger.info("Diarization model loaded successfully")
    return diarize_model


def get_align_model(language_code: str):
    """Load or return cached alignment model for word-level timestamps."""
    global align_model, align_metadata
    if align_model is None:
        logger.info(f"Loading alignment model for language: {language_code}")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language_code,
            device=DEVICE
        )
        logger.info("Alignment model loaded successfully")
    return align_model, align_metadata


class TranscriptionResponse(BaseModel):
    text: str
    task: str = "transcribe"
    language: str
    duration: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None
    segments: Optional[List[Dict[str, Any]]] = None


def format_as_srt(segments: List[Dict]) -> str:
    """Format segments as SRT subtitles."""
    srt_output = []
    for i, segment in enumerate(segments, 1):
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "").strip()
        
        start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
        end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"
        
        speaker = segment.get("speaker", "")
        if speaker:
            text = f"[{speaker}] {text}"
        
        srt_output.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
    
    return "\n".join(srt_output)


def format_as_vtt(segments: List[Dict]) -> str:
    """Format segments as WebVTT subtitles."""
    vtt_output = ["WEBVTT\n"]
    for segment in segments:
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "").strip()
        
        start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{start%60:06.3f}"
        end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{end%60:06.3f}"
        
        speaker = segment.get("speaker", "")
        if speaker:
            text = f"<v {speaker}>{text}"
        
        vtt_output.append(f"{start_time} --> {end_time}\n{text}\n")
    
    return "\n".join(vtt_output)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": ASR_MODEL, "device": DEVICE}


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    # Include multiple model name aliases for LiteLLM compatibility
    models = [
        {
            "id": f"openai/whisper-{ASR_MODEL}",
            "object": "model",
            "created": 1700000000,
            "owned_by": "whisperx"
        },
        {
            "id": f"whisper-{ASR_MODEL}",
            "object": "model",
            "created": 1700000000,
            "owned_by": "whisperx"
        },
        {
            "id": ASR_MODEL,
            "object": "model",
            "created": 1700000000,
            "owned_by": "whisperx"
        },
        {
            "id": "whisper-1",
            "object": "model",
            "created": 1700000000,
            "owned_by": "whisperx"
        }
    ]
    return {
        "object": "list",
        "data": models
    }


@app.post("/v1/audio/transcriptions")
@app.post("/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(default=None),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: Optional[str] = Form(default=None),
    # WhisperX-specific options
    diarize: bool = Form(default=False),
    min_speakers: Optional[int] = Form(default=None),
    max_speakers: Optional[int] = Form(default=None),
    word_timestamps: bool = Form(default=True),
):
    """
    Transcribe audio file (OpenAI-compatible endpoint).
    
    Additional WhisperX options:
    - diarize: Enable speaker diarization
    - min_speakers/max_speakers: Hints for diarization
    - word_timestamps: Enable word-level timestamps
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Load audio
            logger.info(f"Processing audio file: {file.filename}")
            audio = whisperx.load_audio(tmp_path)
            
            # Get model and transcribe
            model_instance = get_whisper_model()
            result = model_instance.transcribe(
                audio,
                batch_size=BATCH_SIZE,
                language=language
            )
            
            detected_language = result.get("language", language or "en")
            
            # Word-level alignment
            if word_timestamps:
                logger.info("Performing word-level alignment...")
                align_model_instance, metadata = get_align_model(detected_language)
                result = whisperx.align(
                    result["segments"],
                    align_model_instance,
                    metadata,
                    audio,
                    DEVICE,
                    return_char_alignments=False
                )
            
            # Speaker diarization
            if diarize and HF_TOKEN:
                logger.info("Performing speaker diarization...")
                diarize_pipeline = get_diarize_model()
                if diarize_pipeline:
                    diarize_segments = diarize_pipeline(
                        audio,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )
                    result = whisperx.assign_word_speakers(diarize_segments, result)
            elif diarize and not HF_TOKEN:
                logger.warning("Diarization requested but HF_TOKEN not set. Skipping...")
            
            # Extract full text
            segments = result.get("segments", [])
            full_text = " ".join([s.get("text", "").strip() for s in segments])
            
            # Format response based on requested format
            if response_format == "text":
                return PlainTextResponse(content=full_text)
            
            elif response_format == "srt":
                return PlainTextResponse(
                    content=format_as_srt(segments),
                    media_type="text/plain"
                )
            
            elif response_format == "vtt":
                return PlainTextResponse(
                    content=format_as_vtt(segments),
                    media_type="text/vtt"
                )
            
            elif response_format == "verbose_json":
                return JSONResponse({
                    "task": "transcribe",
                    "language": detected_language,
                    "duration": len(audio) / 16000,  # Assuming 16kHz sample rate
                    "text": full_text,
                    "segments": segments,
                    "words": [w for s in segments for w in s.get("words", [])]
                })
            
            else:  # json (default)
                return JSONResponse({
                    "text": full_text
                })
        
        finally:
            # Cleanup temp file
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/translations")
@app.post("/audio/translations")
async def translate_audio(
    file: UploadFile = File(...),
    model: str = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    """Translate audio to English (OpenAI-compatible endpoint)."""
    # WhisperX transcribes in the original language, translation would require additional processing
    # For now, transcribe with English output
    return await transcribe_audio(
        file=file,
        model=model,
        language="en",
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        diarize=False,
        word_timestamps=True,
    )


@app.on_event("startup")
async def startup_event():
    """Pre-load models on startup."""
    logger.info("WhisperX ASR Server starting...")
    logger.info(f"Configuration: model={ASR_MODEL}, compute_type={COMPUTE_TYPE}, device={DEVICE}")
    logger.info(f"Diarization available: {bool(HF_TOKEN)}")
    
    # Pre-load the whisper model
    try:
        get_whisper_model()
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
