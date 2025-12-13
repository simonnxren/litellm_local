# API Usage Guide

Complete guide for using the Memoria vLLM service via LiteLLM Python SDK and REST API.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Available Models](#available-models)
- [Embeddings API](#embeddings-api)
- [Chat Completions API](#chat-completions-api)
- [Vision OCR API](#vision-ocr-api)
- [Audio Transcription API](#audio-transcription-api)
- [Streaming Responses](#streaming-responses)
- [Error Handling](#error-handling)
- [Rate Limits & Performance](#rate-limits--performance)

---

## Installation

### Python SDK

```bash
pip install openai
```

The service is compatible with the OpenAI Python SDK.

### Dependencies

```bash
pip install openai httpx python-dotenv
```

---

## Quick Start

### Basic Setup

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8200/v1",  # or your server URL
    api_key="not-needed"  # no authentication required
)
```

### Environment Variables

Create a `.env` file:

```env
OPENAI_API_BASE=http://localhost:8200/v1
OPENAI_API_KEY=not-needed
```

Then in Python:

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY")
)
```

---

## Authentication

**Current Status:** No authentication required.

The service accepts any API key. For production deployments, you can enable LiteLLM's built-in authentication by setting `LITELLM_MASTER_KEY` in the environment.

---

## Available Models

### Check Available Models

```python
models = client.models.list()
for model in models.data:
    print(f"- {model.id}")
```

### Model List

| Model Name | Type | Use Case | Dimensions |
|------------|------|----------|------------|
| `qwen3-embedding-0.6b` | Embeddings | Text embeddings | 1024 |
| `qwen3-4b-thinking-fp8` | Chat | Conversational AI with reasoning | - |
| `hunyuan-ocr` | Vision | Document OCR & extraction | - |
| `whisper-large-v3-turbo` | Audio | Speech transcription | - |

**HuggingFace ID Aliases:**
- `Qwen/Qwen3-Embedding-0.6B`
- `Qwen/Qwen3-4B-Thinking-2507-FP8`
- `tencent/HunyuanOCR`

---

## Embeddings API

Generate vector embeddings for text.

### Single Text Embedding

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8200/v1", api_key="not-needed")

response = client.embeddings.create(
    input="Hello, world!",
    model="qwen3-embedding-0.6b"
)

embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")  # 1024
print(f"Tokens used: {response.usage.total_tokens}")
```

### Batch Embeddings

```python
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming technology",
    "Python is a versatile programming language"
]

response = client.embeddings.create(
    input=texts,
    model="qwen3-embedding-0.6b"
)

for i, data in enumerate(response.data):
    print(f"Text {i}: {len(data.embedding)} dimensions")
```

### RAG Workflow Example

```python
import numpy as np
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8200/v1", api_key="not-needed")

# 1. Embed knowledge base documents
documents = [
    "Renewable energy reduces carbon emissions",
    "Solar panels convert sunlight to electricity",
    "Wind turbines generate clean power"
]

doc_embeddings = []
for doc in documents:
    response = client.embeddings.create(input=doc, model="qwen3-embedding-0.6b")
    doc_embeddings.append(response.data[0].embedding)

# 2. Embed user query
query = "What are the benefits of renewable energy?"
query_response = client.embeddings.create(input=query, model="qwen3-embedding-0.6b")
query_embedding = query_response.data[0].embedding

# 3. Find most similar document (cosine similarity)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
most_similar_idx = np.argmax(similarities)

print(f"Most relevant: {documents[most_similar_idx]}")
print(f"Similarity: {similarities[most_similar_idx]:.3f}")

# 4. Use in chat completion
context = documents[most_similar_idx]
completion = client.chat.completions.create(
    model="qwen3-4b-thinking-fp8",
    messages=[
        {"role": "system", "content": f"Context: {context}"},
        {"role": "user", "content": query}
    ]
)

print(f"\nAnswer: {completion.choices[0].message.content}")
```

### API Response Format

```python
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [0.123, -0.456, ...],  # 1024 floats
            "index": 0
        }
    ],
    "model": "qwen3-embedding-0.6b",
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 5
    }
}
```

---

## Chat Completions API

Generate conversational responses with reasoning capabilities.

### Basic Chat

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8200/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen3-4b-thinking-fp8",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
```

### Reasoning Model (Thinking Mode)

The Qwen3-4B-Thinking model can show its reasoning process:

```python
response = client.chat.completions.create(
    model="qwen3-4b-thinking-fp8",
    messages=[
        {"role": "user", "content": "If x + 5 = 12, what is x?"}
    ],
    max_tokens=500,
    temperature=0.7
)

message = response.choices[0].message

# Reasoning content (thought process)
if hasattr(message, 'reasoning_content') and message.reasoning_content:
    print("🧠 Reasoning:", message.reasoning_content)

# Final answer
print("📝 Answer:", message.content)
```

### Multi-turn Conversation

```python
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is Python?"},
]

response = client.chat.completions.create(
    model="qwen3-4b-thinking-fp8",
    messages=messages
)

# Add assistant response to conversation
messages.append({
    "role": "assistant", 
    "content": response.choices[0].message.content
})

# Continue conversation
messages.append({"role": "user", "content": "What are its main features?"})

response = client.chat.completions.create(
    model="qwen3-4b-thinking-fp8",
    messages=messages
)

print(response.choices[0].message.content)
```

### Advanced Parameters

```python
response = client.chat.completions.create(
    model="qwen3-4b-thinking-fp8",
    messages=[{"role": "user", "content": "Tell me a story"}],
    max_tokens=1000,           # Max response length
    temperature=0.8,           # Creativity (0.0-2.0)
    top_p=0.9,                 # Nucleus sampling
    frequency_penalty=0.5,     # Reduce repetition
    presence_penalty=0.3,      # Encourage new topics
    stop=["\n\n", "END"]      # Stop sequences
)
```

### Structured Output (JSON Schema)

Force the model to respond in a specific JSON format:

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8200/v1", api_key="not-needed")

# Define your schema using Pydantic
class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str
    hobbies: list[str]

response = client.chat.completions.create(
    model="qwen3-4b-thinking-fp8",
    messages=[
        {"role": "user", "content": "Tell me about a software engineer named Alice who is 28 years old"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "schema": PersonInfo.model_json_schema()
        }
    }
)

import json
person = json.loads(response.choices[0].message.content)
print(f"Name: {person['name']}, Age: {person['age']}")
```

### Manual JSON Schema Example

```python
response = client.chat.completions.create(
    model="qwen3-4b-thinking-fp8",
    messages=[
        {"role": "user", "content": "Extract information from: John Doe works as a teacher and loves reading and hiking"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "occupation": {"type": "string"},
                    "interests": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "occupation", "interests"],
                "additionalProperties": False
            }
        }
    }
)

import json
data = json.loads(response.choices[0].message.content)
print(data)  # {"name": "John Doe", "occupation": "teacher", "interests": ["reading", "hiking"]}
```

### API Response Format

```python
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1733875200,
    "model": "qwen3-4b-thinking-fp8",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris.",
                "reasoning_content": null  # or reasoning text
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20
    }
}
```

---

## Vision OCR API

Extract text from images using HunyuanOCR.

### Basic OCR

```python
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8200/v1", api_key="not-needed")

# Load and encode image
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

image_path = "document.png"
base64_image = encode_image(image_path)

# OCR request using chat completions with vision
response = client.chat.completions.create(
    model="hunyuan-ocr",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract all text from this image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    max_tokens=4096
)

extracted_text = response.choices[0].message.content
print(extracted_text)
```

### OCR with Markdown Conversion

```python
response = client.chat.completions.create(
    model="hunyuan-ocr",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Convert this document to markdown format."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        }
    ],
    max_tokens=4096
)

markdown = response.choices[0].message.content
print(markdown)
```

### OCR with Multiple Images

```python
images = ["page1.png", "page2.png", "page3.png"]

for i, img_path in enumerate(images):
    base64_img = encode_image(img_path)
    
    response = client.chat.completions.create(
        model="hunyuan-ocr",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract text from this page."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]
            }
        ],
        max_tokens=4096
    )
    
    print(f"\n--- Page {i+1} ---")
    print(response.choices[0].message.content)
```

### Supported Image Formats

- PNG
- JPEG/JPG
- GIF
- WebP
- BMP

---

## Audio Transcription API

Transcribe audio files using Whisper.

### Basic Transcription

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8200/v1", api_key="not-needed")

audio_file = open("speech.mp3", "rb")

response = client.audio.transcriptions.create(
    model="whisper-large-v3-turbo",
    file=audio_file
)

print(response.text)
```

### Transcription with Language

```python
audio_file = open("speech.mp3", "rb")

response = client.audio.transcriptions.create(
    model="whisper-large-v3-turbo",
    file=audio_file,
    language="en"  # ISO-639-1 code
)

print(response.text)
```

### Advanced Transcription Options

```python
response = client.audio.transcriptions.create(
    model="whisper-large-v3-turbo",
    file=audio_file,
    language="en",
    temperature=0.0,        # Lower = more consistent
    response_format="json"  # json, text, srt, vtt, verbose_json
)

print(response.text)
```

### Supported Audio Formats

- MP3
- MP4
- MPEG
- MPGA
- M4A
- WAV
- WebM
- FLAC
- OGG

### Supported Languages

Whisper supports 99+ languages including:
- `en` - English
- `zh` - Chinese
- `es` - Spanish
- `fr` - French
- `de` - German
- `ja` - Japanese
- `ko` - Korean
- `ru` - Russian
- `pt` - Portuguese
- `ar` - Arabic

---

## Streaming Responses

Stream chat completions token-by-token.

### Streaming Chat

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8200/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="qwen3-4b-thinking-fp8",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

print("Response: ", end="")
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### Streaming with Reasoning

```python
stream = client.chat.completions.create(
    model="qwen3-4b-thinking-fp8",
    messages=[{"role": "user", "content": "Solve: 2x + 3 = 11"}],
    stream=True
)

reasoning_parts = []
content_parts = []

for chunk in stream:
    delta = chunk.choices[0].delta
    
    # Collect reasoning
    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
        reasoning_parts.append(delta.reasoning_content)
    
    # Collect content
    if delta.content is not None:
        content_parts.append(delta.content)
        print(delta.content, end="", flush=True)

print("\n")
if reasoning_parts:
    print("🧠 Reasoning:", "".join(reasoning_parts))
```

### Streaming Completions (Legacy)

```python
stream = client.completions.create(
    model="qwen3-4b-thinking-fp8",
    prompt="Once upon a time",
    max_tokens=100,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].text:
        print(chunk.choices[0].text, end="", flush=True)
print()
```

---

## Error Handling

### Basic Error Handling

```python
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

client = OpenAI(base_url="http://localhost:8200/v1", api_key="not-needed")

try:
    response = client.chat.completions.create(
        model="qwen3-4b-thinking-fp8",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(response.choices[0].message.content)
    
except APIConnectionError as e:
    print(f"Connection error: {e}")
    
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
    
except APIError as e:
    print(f"API error: {e}")
```

### Retry Logic

```python
import time
from openai import OpenAI, APIError

client = OpenAI(base_url="http://localhost:8200/v1", api_key="not-needed")

def api_call_with_retry(max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="qwen3-4b-thinking-fp8",
                messages=[{"role": "user", "content": "Hello"}]
            )
            return response.choices[0].message.content
            
        except APIError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

result = api_call_with_retry()
print(result)
```

### Timeout Configuration

```python
import httpx
from openai import OpenAI

# Custom timeout settings
client = OpenAI(
    base_url="http://localhost:8200/v1",
    api_key="not-needed",
    timeout=httpx.Timeout(60.0, connect=10.0)  # 60s total, 10s connect
)
```

---

## Rate Limits & Performance

### Current Limits

No rate limits enforced by default. Performance depends on:
- GPU availability
- Model size
- Concurrent requests

### Performance Tips

1. **Batch embeddings** instead of individual requests:
```python
# Good: 1 request
texts = ["text1", "text2", "text3"]
response = client.embeddings.create(input=texts, model="qwen3-embedding-0.6b")

# Bad: 3 requests
for text in texts:
    response = client.embeddings.create(input=text, model="qwen3-embedding-0.6b")
```

2. **Use streaming** for long completions:
```python
stream = client.chat.completions.create(
    model="qwen3-4b-thinking-fp8",
    messages=[{"role": "user", "content": prompt}],
    stream=True
)
```

3. **Set appropriate max_tokens**:
```python
# Don't request more than needed
response = client.chat.completions.create(
    model="qwen3-4b-thinking-fp8",
    messages=[{"role": "user", "content": "Brief answer please"}],
    max_tokens=100  # Not 4096
)
```

### Concurrent Requests

```python
import asyncio
from openai import AsyncOpenAI

async def process_batch():
    client = AsyncOpenAI(
        base_url="http://localhost:8200/v1",
        api_key="not-needed"
    )
    
    prompts = ["Question 1?", "Question 2?", "Question 3?"]
    
    tasks = [
        client.chat.completions.create(
            model="qwen3-4b-thinking-fp8",
            messages=[{"role": "user", "content": prompt}]
        )
        for prompt in prompts
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f"Response {i+1}: {response.choices[0].message.content}")

asyncio.run(process_batch())
```


