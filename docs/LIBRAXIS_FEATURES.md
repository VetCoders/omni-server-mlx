# LibraxisAI Contributions

This document describes the features contributed by [LibraxisAI](https://libraxis.ai) to MLX Omni Server.

## Features Overview

### 1. OpenAI Responses API (`/v1/responses`)

Full implementation of OpenAI's Responses API, providing a unified interface for multi-turn conversations with extended capabilities:

- **Multi-turn conversations** with stateful context
- **Multimodal input** (text, images, audio, video)
- **Reasoning/thinking support** with channel parsing
- **Tool/function calling** integration
- **Streaming responses** (SSE)
- **Structured output** with JSON schema

#### Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:10240/v1",
    api_key="not-needed"
)

# Simple text request
response = client.responses.create(
    model="mlx-community/Qwen2.5-7B-Instruct-4bit",
    input="Explain quantum computing in simple terms."
)
print(response.output[0].content[0].text)

# Multimodal request
response = client.responses.create(
    model="llava-1.5",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "What's in this image?"},
            {"type": "input_image", "image_url": "https://example.com/image.png"}
        ]
    }],
    modalities=["text", "image"]
)
```

### 2. Multi-Provider Routing

Intelligent routing across multiple LLM providers with automatic fallback:

- **Round-robin load balancing** within priority tiers
- **Automatic fallback** on provider failure
- **Provider-specific authentication**
- **Configurable retry with exponential backoff**

#### Configuration

```bash
# Environment variables
LLM_BASE_URL=http://localhost:1234              # Primary (LM Studio)
LLM_ALT_BASE_URL=http://localhost:11434         # Secondary (Ollama)
LLM_BASE_URLS=["http://localhost:8100"]         # Additional upstreams

# Cloud providers (if API keys present)
OPENAI_API_KEY=sk-...                           # Fallback to OpenAI
```

#### Programmatic Usage

```python
from mlx_omni_server.routing import MultiProviderRouter, Provider, ProviderType

router = MultiProviderRouter()

# Add local providers
router.add_provider(Provider(
    name="lmstudio",
    base_url="http://localhost:1234",
    provider_type=ProviderType.LMSTUDIO,
    priority=1,
))

router.add_provider(Provider(
    name="ollama",
    base_url="http://localhost:11434",
    provider_type=ProviderType.OLLAMA,
    priority=2,
))

# Request will try LM Studio first, fallback to Ollama on failure
result = await router.call(
    endpoint="/v1/chat/completions",
    payload={"model": "...", "messages": [...]}
)
```

### 3. Circuit Breaker

Protection against cascading failures when providers become unavailable:

- **Three-state machine**: CLOSED → OPEN → HALF_OPEN
- **Automatic recovery** after timeout
- **Configurable thresholds**
- **Per-provider isolation**

#### Usage

```python
from mlx_omni_server.utils.circuit_breaker import CircuitBreaker, get_circuit_breaker

# Get or create named circuit breaker
breaker = get_circuit_breaker(
    name="ollama",
    failure_threshold=5,   # Open after 5 failures
    timeout=60,            # Try recovery after 60s
    success_threshold=2,   # Need 2 successes to close
)

# Use with async functions
try:
    result = await breaker.call(call_provider)
except CircuitBreakerOpen:
    # Handle provider unavailability
    pass
```

### 4. OpenAI Harmony Parser

Support for GPT-OSS models that use the Harmony response format:

- **Automatic detection** of Harmony models
- **Reasoning channel parsing** (analysis/final sections)
- **Output restructuring** for Responses API

```python
from mlx_omni_server.utils.harmony_parser import (
    is_harmony_model,
    parse_reasoning_channels,
    apply_harmony_parsing,
)

# Check if model uses Harmony format
if is_harmony_model("gpt-oss-120b"):
    # Parse reasoning channels
    analysis, final = parse_reasoning_channels(response.reasoning)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MLX Omni Server                            │
├─────────────────────────────────────────────────────────────────┤
│  /v1/responses     │  /v1/chat/completions  │  /anthropic/v1   │
│  (LibraxisAI)      │  (Original)            │  (Original)      │
├────────────────────┴───────────────────────┴───────────────────┤
│                    ResponsesAdapter                             │
│              ↓ normalizes to chat format                        │
├─────────────────────────────────────────────────────────────────┤
│                  MultiProviderRouter                            │
│         ↓ routes with fallback + circuit breaker                │
├─────────────────────────────────────────────────────────────────┤
│  LM Studio   │    Ollama    │   MLX Local   │   Cloud APIs     │
│   :1234      │    :11434    │   (internal)  │   (fallback)     │
└──────────────┴──────────────┴───────────────┴──────────────────┘
```

## Configuration Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BASE_URL` | Primary LLM endpoint | `http://localhost:11434` |
| `LLM_ALT_BASE_URL` | Secondary LLM endpoint | - |
| `LLM_BASE_URLS` | JSON array of additional endpoints | `[]` |
| `OLLAMA_API_URL` | Ollama API endpoint | - |
| `OPENAI_API_KEY` | OpenAI API key (enables cloud fallback) | - |

## Contributing

These features were contributed by the LibraxisAI team:
- Maciej Gad (@Szowesgad)
- Klaudiusz (@gitlaudiusz)

For questions or contributions, visit [LibraxisAI on GitHub](https://github.com/LibraxisAI).

---

**Created by M&K (c)2025 The LibraxisAI Team**
