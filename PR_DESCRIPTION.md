# PR: Add OpenAI Responses API, Multi-Provider Routing, and Circuit Breaker

## Summary

This PR adds enterprise-grade features contributed by **LibraxisAI**, extending MLX Omni Server with production-ready routing, fallback, and API compatibility capabilities.

## New Features

### 1. OpenAI Responses API (`/v1/responses`)

Full implementation of OpenAI's Responses API format, providing:
- Multi-turn conversations with stateful context
- Multimodal input support (text, images, audio, video)
- Reasoning/thinking channel parsing
- Tool/function calling integration
- Streaming responses (SSE)
- Structured output with JSON schema

### 2. Multi-Provider Routing

Intelligent routing across multiple LLM providers:
- Round-robin load balancing within priority tiers
- Automatic fallback on provider failure
- Provider-specific authentication
- Configurable retry with exponential backoff
- Supports: MLX local, Ollama, LM Studio, OpenAI, custom endpoints

### 3. Circuit Breaker

Protection against cascading failures:
- Three-state machine (CLOSED → OPEN → HALF_OPEN)
- Automatic recovery after configurable timeout
- Per-provider isolation
- Thread-safe async implementation

### 4. OpenAI Harmony Parser

Support for GPT-OSS models:
- Automatic detection of Harmony models
- Reasoning channel parsing (analysis/final)
- Output restructuring for Responses API

### 5. Loctree Code Quality Integration

CI/CD integration with Loctree code analysis:
- Pre-commit hooks for dead exports and duplicates
- GitHub Actions workflow for automated analysis
- Health score tracking (current: 64/100)
- Comprehensive analysis documentation

## Files Changed

**New modules:**
- `src/mlx_omni_server/responses/` - Complete Responses API implementation
- `src/mlx_omni_server/routing/` - Multi-provider routing with fallback
- `src/mlx_omni_server/utils/circuit_breaker.py` - Circuit breaker pattern
- `src/mlx_omni_server/utils/harmony_parser.py` - Harmony format parser

**Tests:**
- `tests/test_circuit_breaker.py` - Circuit breaker unit tests
- `tests/test_responses.py` - Responses API tests

**Documentation:**
- `docs/LIBRAXIS_FEATURES.md` - Comprehensive feature documentation
- `docs/LOCTREE_ANALYSIS.md` - Code quality analysis report

**CI/CD:**
- `.github/workflows/loctree.yml` - Loctree code quality workflow
- `.pre-commit-config.yaml` - Extended with Loctree hooks

## Configuration

New environment variables:
```bash
LLM_BASE_URL=http://localhost:1234        # Primary LLM
LLM_ALT_BASE_URL=http://localhost:11434   # Secondary LLM
LLM_BASE_URLS=["http://localhost:8100"]   # Additional upstreams
OPENAI_API_KEY=sk-...                     # Cloud fallback (optional)
```

## Breaking Changes

None. All existing endpoints and behavior are preserved.

## Code Quality

**Loctree Health Score: 64/100**

Known issues (documented in `docs/LOCTREE_ANALYSIS.md`):
- `ResponseFormat` duplicated in 3 files (consolidation recommended)
- `ToolCall` duplicated in 2 files (import from canonical source)
- 12 proper re-exports (not tech debt, good Python practice)

## Test Plan

- [ ] Run `pytest tests/test_circuit_breaker.py` - Circuit breaker tests
- [ ] Run `pytest tests/test_responses.py` - Responses API tests
- [ ] Run `loct auto` - Verify health score
- [ ] Manual test `/v1/responses` endpoint with local model
- [ ] Test multi-provider fallback with multiple backends

---

**Created by M&K (c)2025 The LibraxisAI Team**

Co-Authored-By: Maciej Gad <void@div0.space>
Co-Authored-By: Klaudiusz <the1st@whoai.am>
