# Python Fundamentals for GenAI Developers

## 1. Data Structures: Lists, Tuples, Sets, Dictionaries

### Lists
- **Mutable, ordered collections** of elements
- Can contain mixed types
- Access via index: `list[0]`

```python
# Production example: Batch processing LLM responses
responses = ["response1", "response2", "response3"]
for i, response in enumerate(responses):
    print(f"Batch {i}: {response}")
```

### Tuples
- **Immutable, ordered** collections
- Slightly faster than lists (memory efficient)
- Used for fixed data that shouldn't change

```python
# GenAI context: Storing immutable function signatures
token_counts = (1024, 2048, 4096)  # Supported context windows
```

### Sets
- **Unordered, unique elements**
- No duplicates allowed
- Fast membership testing

```python
# Remove duplicate model names
models = {"gpt-4", "claude-3", "gpt-4", "llama2"}
print(models)  # {'gpt-4', 'claude-3', 'llama2'}
```

### Dictionaries
- **Key-value pairs**, unordered (Python 3.7+, ordered)
- Most commonly used in GenAI (JSON-like)

```python
# GenAI API response structure
response = {
    "id": "msg_123",
    "role": "assistant",
    "content": "Hello, how can I help?",
    "usage": {"input_tokens": 10, "output_tokens": 15}
}
print(response["content"])
```

---

## 2. List vs Tuple

| Aspect | List | Tuple |
|--------|------|-------|
| Mutable | Yes | No |
| Performance | Slower | Faster |
| Use case | Dynamic data | Fixed, immutable data |
| Memory | More | Less |
| Hashable | No (can't use as dict key) | Yes (can use as dict key) |

```python
# GenAI context: Using tuple as dictionary key for cache
embedding_cache = {}
key = ("text", "embedding-model-v1")  # Tuple key
embedding_cache[key] = [0.1, 0.2, 0.3]
```

---

## 3. Mutable vs Immutable Objects

### Immutable
- `int`, `str`, `float`, `tuple`, `frozenset`
- Cannot be changed after creation
- Thread-safe

### Mutable
- `list`, `dict`, `set`
- Can be modified after creation

```python
# GenAI production issue: Shallow copy problem with mutable objects
original_config = {"model": "gpt-4", "temp": 0.7}
copy_config = original_config  # NOT a copy, same reference

copy_config["temp"] = 0.9
print(original_config["temp"])  # 0.9 - PROBLEM!

# Solution: Deep copy
import copy
safe_copy = copy.deepcopy(original_config)
```

---

## 4. List Comprehension

Concise way to create lists. Essential for GenAI data processing.

```python
# Traditional way
responses = []
for response in llm_outputs:
    responses.append(response.strip().lower())

# List comprehension
responses = [r.strip().lower() for r in llm_outputs]

# With condition
long_responses = [r for r in llm_outputs if len(r) > 100]

# Nested: Extract embeddings from multiple documents
embeddings = [
    embedding
    for doc in documents
    for embedding in doc.get_embeddings()
]

# GenAI example: Filter tokens above threshold
valid_tokens = [t for t in tokens if t.confidence > 0.8]
```

---

## 5. *args and **kwargs

### *args (Non-keyword Arguments)
- Variable number of positional arguments
- Packed as a tuple

```python
def process_llm_responses(*responses):
    """Handle variable number of LLM responses"""
    for response in responses:
        print(f"Processing: {response}")

process_llm_responses("response1", "response2", "response3")
```

### **kwargs (Keyword Arguments)
- Variable number of named arguments
- Packed as a dictionary

```python
def create_llm_request(**params):
    """Flexible LLM request builder"""
    print(params)
    # {'model': 'gpt-4', 'temperature': 0.7, 'max_tokens': 1000}

create_llm_request(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

# GenAI production: Dynamic config
def configure_agent(*tools, **settings):
    agent_config = {
        "tools": tools,
        "max_iterations": settings.get("max_iterations", 10),
        "timeout": settings.get("timeout", 30)
    }
    return agent_config
```

---

## 6. Lambda Functions

Anonymous functions. Useful for simple operations in GenAI pipelines.

```python
# Sort documents by relevance score
documents = [
    {"text": "doc1", "score": 0.8},
    {"text": "doc2", "score": 0.95},
    {"text": "doc3", "score": 0.7}
]

sorted_docs = sorted(documents, key=lambda x: x["score"], reverse=True)

# Map transformation: Extract tokens from responses
responses = ["Hello world", "How are you"]
token_lists = list(map(lambda r: r.split(), responses))
```

---

## 7. Decorators

Functions that modify other functions. Critical for GenAI production code.

```python
import time
from functools import wraps

# Caching decorator for embedding calls
def cache_embeddings(func):
    cache = {}
    
    @wraps(func)
    def wrapper(text):
        if text in cache:
            return cache[text]
        result = func(text)
        cache[text] = result
        return result
    return wrapper

@cache_embeddings
def get_embedding(text):
    # Expensive embedding API call
    return [0.1, 0.2, 0.3]

# Retry decorator for LLM API failures
def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

@retry_on_failure(max_retries=3)
def call_llm_api(prompt):
    # API call that might fail
    pass
```

---

## 8. Generators

Memory-efficient way to produce values lazily. Essential for processing large documents.

```python
# Traditional: Load all chunks into memory
def get_all_chunks(document):
    chunks = []
    for i in range(0, len(document), chunk_size):
        chunks.append(document[i:i+chunk_size])
    return chunks

# Generator: Memory efficient
def chunk_generator(document, chunk_size=1000):
    """Yield chunks one at a time"""
    for i in range(0, len(document), chunk_size):
        yield document[i:i+chunk_size]

# GenAI production: Stream LLM tokens
def stream_llm_response(prompt):
    """Generator for streaming response tokens"""
    for token in llm.stream(prompt):
        yield token

# Usage
for chunk in chunk_generator(large_document):
    process_chunk_with_embedding(chunk)  # One chunk at a time
```

---

## 9. Iterator vs Iterable

- **Iterable**: Object with `__iter__()` (returns iterator)
- **Iterator**: Object with `__next__()` and `__iter__()`

```python
# List is iterable but not an iterator
my_list = [1, 2, 3]
iterator = iter(my_list)  # Get iterator
print(next(iterator))  # 1
print(next(iterator))  # 2

# GenAI: Custom iterator for token stream
class TokenIterator:
    def __init__(self, tokens):
        self.tokens = tokens
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.tokens):
            raise StopIteration
        token = self.tokens[self.index]
        self.index += 1
        return token

tokens = TokenIterator(["Hello", "world", "!"])
for token in tokens:
    print(token)
```

---

## 10. Shallow Copy vs Deep Copy

Critical for preventing bugs in GenAI pipelines.

```python
import copy

# Shallow copy: Only copies top level
original = {
    "config": {"model": "gpt-4", "params": [0.7, 0.8]},
    "name": "agent1"
}

shallow = copy.copy(original)
shallow["config"]["model"] = "gpt-3.5"
print(original["config"]["model"])  # "gpt-3.5" - AFFECTED!

# Deep copy: Recursively copies everything
deep = copy.deepcopy(original)
deep["config"]["model"] = "claude-3"
print(original["config"]["model"])  # Still "gpt-4" - SAFE!

# GenAI context: Agent state management
agent_state = {
    "memory": [{"role": "user", "content": "Hi"}],
    "tools": ["search", "calculator"]
}

# Wrong approach (shallow copy)
backup_state = copy.copy(agent_state)

# Correct approach (deep copy)
safe_backup = copy.deepcopy(agent_state)
```

---

## 11. Exception Handling in Python

Essential for production GenAI systems.

```python
import requests

def call_llm_api(prompt):
    try:
        response = requests.post(
            "https://api.openai.com/v1/messages",
            json={"prompt": prompt}
        )
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.Timeout:
        print("API call timed out")
        # Implement fallback logic
        return get_cached_response(prompt)
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("Rate limited")
            # Handle rate limiting
        elif e.response.status_code == 401:
            print("Authentication failed")
        raise
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
    
    finally:
        print("API call completed")

# Custom exception for GenAI
class EmbeddingError(Exception):
    """Raised when embedding generation fails"""
    pass

class RAGError(Exception):
    """Raised when RAG pipeline fails"""
    pass
```

---

## 12. async/await

Essential for handling multiple LLM API calls concurrently.

```python
import asyncio
import aiohttp

async def call_llm_async(prompt):
    """Single async LLM call"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/messages",
            json={"prompt": prompt}
        ) as response:
            return await response.json()

async def process_multiple_prompts(prompts):
    """Process multiple prompts concurrently"""
    tasks = [call_llm_async(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results

# GenAI production: Concurrent embedding generation
async def generate_embeddings_batch(texts):
    """Generate embeddings for multiple texts concurrently"""
    tasks = [
        get_embedding_async(text)
        for text in texts
    ]
    embeddings = await asyncio.gather(*tasks)
    return embeddings

# Run async function
if __name__ == "__main__":
    prompts = ["What is AI?", "Explain RAG", "What is LLM?"]
    results = asyncio.run(process_multiple_prompts(prompts))
```

---

## 13. Multithreading vs Multiprocessing

| Feature | Multithreading | Multiprocessing |
|---------|---|---|
| Use case | I/O-bound (API calls) | CPU-bound |
| GIL | Affected (Python) | Independent processes |
| Memory | Low | High |
| Speed | Better for I/O | Better for computation |

```python
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Multithreading: Good for API calls
def call_multiple_llms_threaded(prompts):
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(call_llm_api, prompts))
    return results

# Multiprocessing: Good for embedding generation (CPU-intensive)
def generate_embeddings_parallel(texts):
    with ProcessPoolExecutor(max_workers=4) as executor:
        embeddings = list(executor.map(compute_embedding, texts))
    return embeddings
```

---

## 14. Making API Calls with Python

```python
import requests
import json

# Simple GET
response = requests.get("https://api.example.com/models")
data = response.json()

# POST with headers and payload
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers=headers,
    json=payload,
    timeout=30
)

# Error handling
if response.status_code == 200:
    result = response.json()
elif response.status_code == 429:
    print("Rate limited")
elif response.status_code == 401:
    print("Unauthorized")

# Using requests with retry
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

response = session.post(url, json=payload)
```

---

## 15. Parsing JSON

```python
import json

# Parse JSON string
json_str = '{\"model\": \"gpt-4\", \"temperature\": 0.7}'
data = json.loads(json_str)
print(data["model"])  # gpt-4

# Convert Python dict to JSON string
config = {"model": "gpt-4", "temperature": 0.7}
json_str = json.dumps(config, indent=2)

# From API response
response = requests.get(url)
data = response.json()  # Automatically parses JSON

# Handling nested JSON (GenAI API responses)
response_data = {
    "choices": [
        {
            "message": {
                "content": "Response text",
                "role": "assistant"
            }
        }
    ]
}

content = response_data["choices"][0]["message"]["content"]

# Using json.loads with error handling
try:
    data = json.loads(raw_response)
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
```

---

## 16. Building REST API with FastAPI

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI(title="GenAI API")

# Data models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "gpt-4"
    temperature: float = 0.7

class ChatResponse(BaseModel):
    id: str
    content: str
    model: str

# Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send messages to LLM"""
    try:
        response = await call_llm_async(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature
        )
        return ChatResponse(
            id="msg_123",
            content=response["content"],
            model=request.model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """Get available models"""
    return {"models": ["gpt-4", "gpt-3.5", "claude-3"]}

@app.post("/embedding")
async def get_embedding(text: str):
    """Generate embedding for text"""
    embedding = await get_embedding_async(text)
    return {"text": text, "embedding": embedding}

# Background tasks for long operations
@app.post("/batch-process")
async def batch_process(texts: List[str], background_tasks: BackgroundTasks):
    background_tasks.add_task(process_texts_async, texts)
    return {"status": "processing"}

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 17. Handling API Failures and Retries

```python
import time
from typing import Callable, TypeVar, Any
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 32.0,
    exponential_base: float = 2.0
) -> Any:
    """
    Retry logic with exponential backoff for API calls
    Used in production GenAI systems for handling transient failures
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed after {max_retries} attempts: {e}")
                raise
            
            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            jitter = delay * 0.1 * (2 * random.random() - 1)
            total_delay = delay + jitter
            
            logger.warning(
                f"Attempt {attempt + 1} failed. "
                f"Retrying in {total_delay:.2f}s: {e}"
            )
            time.sleep(total_delay)

# Usage in GenAI production
def fetch_llm_response(prompt: str):
    return retry_with_exponential_backoff(
        lambda: call_llm_api(prompt),
        max_retries=3
    )

# With specific exception handling
def robust_api_call(url: str, payload: dict):
    def _call():
        response = requests.post(url, json=payload, timeout=10)
        
        # Handle specific errors
        if response.status_code == 429:  # Rate limited
            raise RateLimitError("Too many requests")
        elif response.status_code == 503:  # Service unavailable
            raise ServiceUnavailableError("Service temporarily down")
        elif response.status_code >= 500:
            raise ServerError(f"Server error: {response.status_code}")
        
        response.raise_for_status()
        return response.json()
    
    return retry_with_exponential_backoff(_call, max_retries=5)
```

---

## 18. Structuring a Python GenAI Project

```
genai-project/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py          # LLM client (OpenAI, Claude, etc.)
│   │   ├── models.py          # Data models
│   │   └── prompts.py         # Prompt templates
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_loader.py # Document ingestion
│   │   ├── chunker.py         # Document chunking
│   │   ├── embedder.py        # Embedding generation
│   │   └── retriever.py       # Retrieval logic
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py      # Base agent class
│   │   ├── tools.py           # Agent tools
│   │   └── memory.py          # Agent memory
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py          # Logging setup
│   │   ├── cache.py           # Caching utilities
│   │   └── validators.py      # Input validation
│   └── api/
│       ├── __init__.py
│       └── routes.py          # FastAPI routes
├── tests/
│   ├── test_llm.py
│   ├── test_rag.py
│   └── test_agents.py
├── notebooks/
│   └── exploration.ipynb      # Development notebooks
├── config/
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
├── requirements.txt
├── .env.example
├── main.py                    # Entry point
└── README.md
```

### Example Implementation

```python
# src/config.py
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # LLM Configuration
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    
    # RAG Configuration
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Vector Database
    vector_db_url: str = "http://localhost:6333"
    
    class Config:
        env_file = ".env"

settings = Settings()

# src/llm/client.py
from src.config import settings
import openai

class LLMClient:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
    
    async def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )
        return response.choices[0].message.content

# main.py
from fastapi import FastAPI
from src.api.routes import router
from src.config import settings

app = FastAPI(title="GenAI Application")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Summary

- **Data structures**: Lists, tuples, sets, dicts are foundational
- **Comprehensions**: Make code concise and Pythonic
- ***args/**kwargs**: Enable flexible function signatures
- **Decorators**: Essential for caching, retrying, logging
- **Generators**: Memory-efficient for large datasets
- **Async/await**: Critical for concurrent API calls
- **Exception handling**: Robust production code
- **API integration**: Use requests with retry logic
- **FastAPI**: Modern framework for GenAI services
- **Project structure**: Organized, maintainable code

In GenAI production, you'll spend 80% of time on API integration, caching, error handling, and async operations.
