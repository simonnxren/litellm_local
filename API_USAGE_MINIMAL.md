# LiteLLM API Usage

**Gateway (Local)**: `http://localhost:8200`  
**Gateway (LAN)**: `http://<YOUR_IP>:8200` (e.g., `http://192.168.1.100:8200`)  
**Models**: `qwen3-embedding-0.6b`, `qwen3-8b-fp8`, `hunyuan-ocr`

> **LAN Access**: Services are accessible from any device on your local network. Replace `localhost` with your server's IP address.

---

## 1. Embeddings

```python
import openai

# Local access
client = openai.OpenAI(
    base_url="http://localhost:8200/v1",
    api_key="dummy"
)

# LAN access (replace with your server IP)
# client = openai.OpenAI(
#     base_url="http://192.168.1.100:8200/v1",
#     api_key="dummy"
# )

# Single text
response = client.embeddings.create(
    model="qwen3-embedding-0.6b",
    input="Your text here"
)
embedding = response.data[0].embedding  # 1024 dimensions

# Batch texts
response = client.embeddings.create(
    model="qwen3-embedding-0.6b",
    input=["text1", "text2", "text3"]
)
```

**cURL (Local)**:
```bash
curl http://localhost:8200/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": "Your text"}'
```

**cURL (LAN)**:
```bash
# Replace 192.168.1.100 with your server IP
curl http://192.168.1.100:8200/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": "Your text"}'
```

---

## 2. Chat Completions

```python
# Basic
response = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    max_tokens=512,
    temperature=0.7
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[{"role": "user", "content": "Write a poem"}],
    max_tokens=256,
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
```

**cURL**:
```bash
curl http://localhost:8200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-fp8",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 100
  }'
```

**Context**: 20,480 tokens max (input + output combined)

---

## 3. OCR (Vision)

```python
import base64

# Read image
with open("document.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

# OCR
response = client.chat.completions.create(
    model="hunyuan-ocr",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract all text from this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            }
        ]
    }],
    max_tokens=2048
)
print(response.choices[0].message.content)
```

**cURL**:
```bash
IMAGE_B64=$(base64 -w 0 your_image.jpg)
curl http://localhost:8200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hunyuan-ocr",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Extract text"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'$IMAGE_B64'"}}
      ]
    }],
    "max_tokens": 2048
  }'
```

---

## Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `temperature` | 0.0-1.0 | 0.1-0.3 factual, 0.7 balanced, 0.8-1.0 creative |
| `max_tokens` | 1-20480 | Max output tokens (chat context is 20,480 total) |
| `top_p` | 0.0-1.0 | Nucleus sampling (default 0.9) |
| `stream` | true/false | Enable streaming responses |
