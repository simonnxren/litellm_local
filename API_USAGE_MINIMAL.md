# API Usage Guide - Minimal Setup

**Memoria vLLM Minimal Setup**: Embeddings + Chat Completions (32K context)

This guide covers the minimal deployment with only 2 services:
- **Embeddings**: Qwen3-Embedding-0.6B (1024 dimensions)
- **Chat**: Qwen3-8B-FP8 (32K context, 8B parameters)

## Installation

```bash
pip install openai python-dotenv
```

## Quick Start

```python
from openai import OpenAI

# Connect to the gateway
client = OpenAI(
    base_url="http://localhost:8200/v1",
    api_key="not-needed"  # No authentication required
)

# Generate embeddings
response = client.embeddings.create(
    model="qwen3-embedding-0.6b",
    input="Hello, world!"
)
print(f"Embedding dimensions: {len(response.data[0].embedding)}")

# Chat completion (clean output with /no_think)
chat = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[
        {"role": "user", "content": "What is the capital of France? /no_think"}
    ],
    max_tokens=100
)
print(chat.choices[0].message.content)
```

---

## 1. Embeddings

### Basic Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8200/v1",
    api_key="not-needed"
)

# Single text
response = client.embeddings.create(
    model="qwen3-embedding-0.6b",
    input="The quick brown fox jumps over the lazy dog"
)

embedding = response.data[0].embedding
print(f"Embedding: {len(embedding)} dimensions")  # 1024 dimensions
```

### Batch Embeddings

```python
# Multiple texts in one request
texts = [
    "First document about machine learning",
    "Second document about artificial intelligence",
    "Third document about neural networks"
]

response = client.embeddings.create(
    model="qwen3-embedding-0.6b",
    input=texts
)

embeddings = [item.embedding for item in response.data]
print(f"Generated {len(embeddings)} embeddings")

# Each embedding is 1024-dimensional
for i, emb in enumerate(embeddings):
    print(f"Document {i+1}: {len(emb)} dimensions")
```

### Using with Vector Databases

```python
import numpy as np
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8200/v1", api_key="dummy")

def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text."""
    response = client.embeddings.create(
        model="qwen3-embedding-0.6b",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

# Example: Semantic search
query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Python is a programming language",
    "Deep learning uses neural networks"
]

query_emb = get_embedding(query)
doc_embeddings = [get_embedding(doc) for doc in documents]

# Find most similar document
similarities = [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embeddings]
best_match_idx = similarities.index(max(similarities))

print(f"Query: {query}")
print(f"Best match: {documents[best_match_idx]}")
print(f"Similarity: {similarities[best_match_idx]:.4f}")
```

---

## 2. Chat Completions

### Understanding Thinking Mode

Qwen3-8B-FP8 has **thinking mode enabled by default**. This means:
- The model generates internal reasoning in `<think>...</think>` blocks
- Useful for complex reasoning, math, and coding tasks
- Can be disabled per-request for faster, cleaner responses

### Clean Chat (No Thinking)

For simple queries, add `/no_think` to get clean output without thinking tags:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8200/v1", api_key="dummy")

# Clean response without thinking
response = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[
        {"role": "user", "content": "What is the capital of Japan? /no_think"}
    ],
    max_tokens=50,
    temperature=0.7
)

print(response.choices[0].message.content)
# Output: "Tokyo"
```

### Chat with Thinking (Complex Reasoning)

For complex tasks, omit `/no_think` to get detailed reasoning:

```python
response = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[
        {"role": "user", "content": "If a train travels 120 km in 2 hours, what is its average speed in m/s?"}
    ],
    max_tokens=500
)

# Response will include <think>...</think> block with reasoning
print(response.choices[0].message.content)
```

### Multi-Turn Conversations

```python
# Conversation history
messages = [
    {"role": "user", "content": "I have 5 apples /no_think"},
    {"role": "assistant", "content": "You have 5 apples."},
    {"role": "user", "content": "I eat 2, then buy 3 more. How many do I have now? /no_think"}
]

response = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=messages,
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
# Output: "You now have 6 apples."
```

### Streaming Responses

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8200/v1", api_key="dummy")

stream = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[
        {"role": "user", "content": "Write a haiku about coding /no_think"}
    ],
    max_tokens=100,
    stream=True
)

print("Response: ", end="", flush=True)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### System Prompts

```python
response = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[
        {"role": "system", "content": "You are a helpful Python programming assistant. Provide concise answers."},
        {"role": "user", "content": "How do I read a JSON file in Python? /no_think"}
    ],
    max_tokens=200,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Temperature and Sampling Parameters

```python
# Creative writing (higher temperature)
creative = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[{"role": "user", "content": "Write a creative opening line for a sci-fi story /no_think"}],
    max_tokens=100,
    temperature=0.9,  # More random/creative
    top_p=0.95
)

# Factual responses (lower temperature)
factual = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[{"role": "user", "content": "What is the speed of light? /no_think"}],
    max_tokens=50,
    temperature=0.3,  # More deterministic
    top_p=0.8
)
```

**Recommended Settings:**
- **With `/no_think`**: `temperature=0.7`, `top_p=0.8`
- **Without `/no_think`** (thinking mode): `temperature=0.6`, `top_p=0.95`

---

## 3. Structured Outputs (JSON Schema)

Perfect for Graphiti knowledge graphs and structured data extraction.

### Using Pydantic Models

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(base_url="http://localhost:8200/v1", api_key="dummy")

class Person(BaseModel):
    name: str
    age: int
    occupation: str
    city: str

# Extract structured data with /no_think for clean output
response = client.beta.chat.completions.parse(
    model="qwen3-8b-fp8",
    messages=[
        {
            "role": "user",
            "content": "Extract information: Alice is a 30-year-old software engineer living in San Francisco. /no_think"
        }
    ],
    response_format=Person,
)

person = response.choices[0].message.parsed
print(f"Name: {person.name}")
print(f"Age: {person.age}")
print(f"Job: {person.occupation}")
print(f"City: {person.city}")
```

### Complex Nested Structures

```python
from pydantic import BaseModel
from typing import List

class Entity(BaseModel):
    name: str
    type: str  # "person", "organization", "location", etc.

class Relationship(BaseModel):
    source: str
    target: str
    relation: str  # "works_at", "located_in", etc.

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

response = client.beta.chat.completions.parse(
    model="qwen3-8b-fp8",
    messages=[
        {
            "role": "user",
            "content": """
Extract entities and relationships:
Alice works at Google in Mountain View. Bob is her colleague and lives in San Jose.
/no_think
            """.strip()
        }
    ],
    response_format=KnowledgeGraph,
)

kg = response.choices[0].message.parsed
print(f"Entities: {len(kg.entities)}")
for entity in kg.entities:
    print(f"  - {entity.name} ({entity.type})")

print(f"Relationships: {len(kg.relationships)}")
for rel in kg.relationships:
    print(f"  - {rel.source} --[{rel.relation}]--> {rel.target}")
```

### Manual JSON Schema

```python
response = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[
        {
            "role": "user",
            "content": "Extract: Product name is iPhone 15, price $999, in stock. /no_think"
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "product_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "number"},
                    "in_stock": {"type": "boolean"}
                },
                "required": ["name", "price", "in_stock"],
                "additionalProperties": False
            }
        }
    },
    max_tokens=100
)

import json
product = json.loads(response.choices[0].message.content)
print(f"Product: {product['name']}, Price: ${product['price']}, In Stock: {product['in_stock']}")
```

---

## 4. Graphiti Integration

Using with [Graphiti](https://github.com/getzep/graphiti) for knowledge graph management:

```python
from openai import OpenAI
from graphiti import Graphiti
from pydantic import BaseModel
from typing import List

# Initialize OpenAI client pointing to memoria_vllm
client = OpenAI(
    base_url="http://localhost:8200/v1",
    api_key="not-needed"
)

# Initialize Graphiti with custom LLM
graphiti = Graphiti(
    llm_client=client,
    embedding_client=client,
    llm_model="qwen3-8b-fp8",
    embedding_model="qwen3-embedding-0.6b"
)

# Define entity extraction schema
class Entity(BaseModel):
    name: str
    type: str
    description: str

class Triple(BaseModel):
    subject: str
    predicate: str
    object: str

# Extract entities with /no_think for cleaner extraction
text = """
John Smith works as a Senior Engineer at TechCorp in Seattle.
He reports to Jane Doe, who is the Engineering Director.
TechCorp was founded in 2010 and specializes in cloud computing.
"""

response = client.beta.chat.completions.parse(
    model="qwen3-8b-fp8",
    messages=[
        {
            "role": "system",
            "content": "Extract all entities and relationships from the text."
        },
        {
            "role": "user",
            "content": f"{text}\n/no_think"
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "knowledge_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["name", "type", "description"],
                            "additionalProperties": False
                        }
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "predicate": {"type": "string"},
                                "object": {"type": "string"}
                            },
                            "required": ["subject", "predicate", "object"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["entities", "relationships"],
                "additionalProperties": False
            }
        }
    }
)

import json
extracted = json.loads(response.choices[0].message.content)
print(f"Extracted {len(extracted['entities'])} entities")
print(f"Extracted {len(extracted['relationships'])} relationships")
```

---

## 5. Long Context Processing (32K Tokens)

The Qwen3-8B-FP8 model supports up to **32,768 tokens** of context (approximately 24,000 words).

### Processing Long Documents

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8200/v1", api_key="dummy")

# Example: Summarize a long document
with open("long_document.txt", "r") as f:
    document = f.read()

response = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes long documents."
        },
        {
            "role": "user",
            "content": f"Summarize the key points from this document:\n\n{document}\n\n/no_think"
        }
    ],
    max_tokens=500,
    temperature=0.5
)

print(response.choices[0].message.content)
```

### Counting Tokens

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Estimate token count (Qwen uses similar tokenization)."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

document = "Your long document text here..."
token_count = count_tokens(document)

print(f"Document has approximately {token_count} tokens")
if token_count > 32000:
    print("Warning: Document may exceed 32K token limit")
else:
    print(f"Safe to process ({32768 - token_count} tokens remaining)")
```

---

## 6. Error Handling

```python
from openai import OpenAI, OpenAIError, APIError, RateLimitError, APIConnectionError

client = OpenAI(base_url="http://localhost:8200/v1", api_key="dummy")

def safe_chat(prompt: str, max_retries: int = 3) -> str:
    """Chat with automatic retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="qwen3-8b-fp8",
                messages=[{"role": "user", "content": f"{prompt} /no_think"}],
                max_tokens=200,
                timeout=30.0
            )
            return response.choices[0].message.content
        
        except APIConnectionError as e:
            print(f"Connection error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
        
        except RateLimitError as e:
            print(f"Rate limit hit: {e}")
            time.sleep(5)
        
        except APIError as e:
            print(f"API error: {e}")
            raise
        
        except OpenAIError as e:
            print(f"OpenAI error: {e}")
            raise

# Usage
try:
    result = safe_chat("What is Python?")
    print(result)
except Exception as e:
    print(f"Failed after retries: {e}")
```

---

## 7. Performance Tips

### Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8200/v1", api_key="dummy")

def process_text(text: str) -> dict:
    """Process a single text with embedding and summary."""
    # Get embedding
    emb_response = client.embeddings.create(
        model="qwen3-embedding-0.6b",
        input=text
    )
    
    # Get summary
    chat_response = client.chat.completions.create(
        model="qwen3-8b-fp8",
        messages=[{"role": "user", "content": f"Summarize in one sentence: {text} /no_think"}],
        max_tokens=50
    )
    
    return {
        "text": text,
        "embedding": emb_response.data[0].embedding,
        "summary": chat_response.choices[0].message.content
    }

# Process multiple texts in parallel
texts = [
    "Text about machine learning...",
    "Text about data science...",
    "Text about artificial intelligence..."
]

with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(process_text, texts))

for i, result in enumerate(results):
    print(f"{i+1}. {result['summary']}")
```

### Optimizing Token Usage

```python
# Instead of this (wastes tokens):
response = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[
        {"role": "user", "content": "Can you please tell me what is the capital of France? I would really appreciate it if you could answer this question for me. /no_think"}
    ],
    max_tokens=100
)

# Do this (concise):
response = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[
        {"role": "user", "content": "Capital of France? /no_think"}
    ],
    max_tokens=20
)
```

---

## 8. Example: Complete RAG Pipeline

```python
from openai import OpenAI
import numpy as np
from typing import List, Tuple

client = OpenAI(base_url="http://localhost:8200/v1", api_key="dummy")

class SimpleRAG:
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_document(self, text: str):
        """Add a document to the knowledge base."""
        response = client.embeddings.create(
            model="qwen3-embedding-0.6b",
            input=text
        )
        self.documents.append(text)
        self.embeddings.append(response.data[0].embedding)
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for relevant documents."""
        query_response = client.embeddings.create(
            model="qwen3-embedding-0.6b",
            input=query
        )
        query_emb = np.array(query_response.data[0].embedding)
        
        # Calculate similarities
        similarities = []
        for doc, emb in zip(self.documents, self.embeddings):
            emb_array = np.array(emb)
            similarity = np.dot(query_emb, emb_array) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb_array)
            )
            similarities.append((doc, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def answer(self, question: str) -> str:
        """Answer a question using RAG."""
        # Find relevant documents
        relevant_docs = self.search(question, top_k=3)
        context = "\n\n".join([doc for doc, _ in relevant_docs])
        
        # Generate answer
        response = client.chat.completions.create(
            model="qwen3-8b-fp8",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question based only on the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\n/no_think"
                }
            ],
            max_tokens=200,
            temperature=0.5
        )
        
        return response.choices[0].message.content

# Usage
rag = SimpleRAG()
rag.add_document("Python is a high-level programming language.")
rag.add_document("Machine learning is a subset of AI.")
rag.add_document("Neural networks are inspired by biological neurons.")

answer = rag.answer("What is Python?")
print(answer)
```

---

## API Reference Summary

### Embeddings Endpoint
- **Model**: `qwen3-embedding-0.6b`
- **Dimensions**: 1024
- **Max tokens per request**: ~8192
- **Use case**: Semantic search, clustering, classification

### Chat Completions Endpoint
- **Model**: `qwen3-8b-fp8`
- **Context length**: 32,768 tokens (~24,000 words)
- **Parameters**: 8 billion
- **Thinking mode**: Enabled by default (add `/no_think` to disable)
- **Structured outputs**: JSON Schema supported
- **Temperature**: 0.7 (no_think), 0.6 (thinking mode)
- **Use cases**: Chat, reasoning, code generation, knowledge extraction

### Model Aliases
Both models support their full HuggingFace IDs:
- `Qwen/Qwen3-Embedding-0.6B`
- `Qwen/Qwen3-8B-FP8`

---

## Support & Resources

- **Repository**: [github.com/simonnxren/memoria_vllm](https://github.com/simonnxren/memoria_vllm)
- **Qwen3 Docs**: [qwen.readthedocs.io](https://qwen.readthedocs.io)
- **OpenAI Python SDK**: [github.com/openai/openai-python](https://github.com/openai/openai-python)
- **Graphiti**: [github.com/getzep/graphiti](https://github.com/getzep/graphiti)

---

**Last Updated**: 2025-12-10  
**Model Versions**: Qwen3-Embedding-0.6B, Qwen3-8B-FP8  
**Deployment**: Minimal (Embeddings + Chat only)
