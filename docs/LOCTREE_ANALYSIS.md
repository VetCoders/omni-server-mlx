# Loctree Code Quality Analysis

**Date:** 2025-12-23
**Health Score:** 64/100
**Files Analyzed:** 99
**Total LOC:** 14,227

## Executive Summary

This analysis identifies code quality issues and provides actionable recommendations for improving the codebase health score.

## Findings Summary

| Category | Count | Severity |
|----------|-------|----------|
| Same-Language Twins | 15 | Medium |
| Dead Exports | 1 | Low (False Positive) |
| Circular Imports | 0 | N/A |
| Cross-Language Twins | 0 | N/A |

## Detailed Analysis

### 1. Same-Language Twins (Duplicates)

These 15 findings break down into two categories:

#### A. Proper Re-exports (Not Tech Debt - 12 items)

These are **intentional** re-exports in `__init__.py` files for cleaner imports:

| Symbol | Module | Re-exported From |
|--------|--------|------------------|
| `Provider` | `routing/__init__.py` | `multi_provider.py` |
| `ProviderType` | `routing/__init__.py` | `multi_provider.py` |
| `MultiProviderRouter` | `routing/__init__.py` | `multi_provider.py` |
| `UpstreamPool` | `routing/__init__.py` | `upstream_pool.py` |
| `UpstreamTarget` | `routing/__init__.py` | `upstream_pool.py` |
| `get_default_router` | `routing/__init__.py` | `multi_provider.py` |
| `ResponseRequest` | `responses/__init__.py` | `schema.py` |
| `ResponseResponse` | `responses/__init__.py` | `schema.py` |
| `apply_harmony_parsing` | `utils/harmony_parser.py` | Re-export pattern |
| `build_harmony_output_entries` | `utils/harmony_parser.py` | Re-export pattern |
| `create_harmony_conversation` | `utils/harmony_parser.py` | Re-export pattern |
| `is_harmony_model` | `utils/harmony_parser.py` | Re-export pattern |
| `parse_reasoning_channels` | `utils/harmony_parser.py` | Re-export pattern |

**Recommendation:** No action needed. This is a good Python practice.

#### B. Actual Duplicates (Tech Debt - 3 items)

These are genuine duplications that should be consolidated:

##### 1. `ResponseFormat` (Priority: High)

**Locations:**
- `src/mlx_omni_server/stt/schema.py:8` (canonical, 1 import)
- `src/mlx_omni_server/images/schema.py:25` (1 import)
- `src/mlx_omni_server/chat/openai/schema.py:179` (0 imports)

**Recommendation:** Create a shared schema module:
```python
# src/mlx_omni_server/common/schema.py
class ResponseFormat(Enum):
    ...

# Then import from common in all locations
```

##### 2. `ToolCall` (Priority: Medium)

**Locations:**
- `src/mlx_omni_server/chat/mlx/core_types.py:8` (canonical, 10 imports)
- `src/mlx_omni_server/chat/openai/schema.py:48` (1 import)

**Recommendation:** Use the canonical definition from `core_types.py`:
```python
# In chat/openai/schema.py, replace local class with:
from ..mlx.core_types import ToolCall
```

##### 3. `get_models_service` (Priority: Low)

**Locations:**
- `src/mlx_omni_server/chat/anthropic/router.py:22`
- `src/mlx_omni_server/chat/openai/models/models.py:12`

**Note:** These are different factory functions for different providers - not actual duplicates.

**Recommendation:** Rename for clarity:
- `get_anthropic_models_service()` in anthropic/router.py
- `get_openai_models_service()` in openai/models/models.py

### 2. Dead Export (False Positive)

**Symbol:** `AnthropicMessagesAdapter`
**Location:** `src/mlx_omni_server/chat/anthropic/anthropic_messages_adapter.py:34`

**Analysis:** This is a **false positive**. The class is used internally:
- Imported in `chat/anthropic/router.py:8`
- Instantiated in `chat/anthropic/router.py:110`

Loctree flagged it because there's no `__init__.py` re-export, but internal usage is correct.

**Recommendation:** No action needed, or optionally add to `__init__.py` for explicit public API.

### 3. Lazy Cycle (Info)

**Members:**
- `src/mlx_omni_server/chat/mlx/wrapper_cache.py`
- `src/mlx_omni_server/chat/mlx/chat_generator.py`

**Note:** This is a "lazy" cycle (import inside function, not at module level), which is a valid Python pattern to avoid circular import errors.

**Recommendation:** No action needed. This is intentional design.

## Health Score Improvement Plan

| Action | Impact on Score |
|--------|----------------|
| Consolidate `ResponseFormat` | +5 |
| Fix `ToolCall` import | +3 |
| Rename `get_models_service` variants | +2 |
| **Total Potential** | **~74/100** |

## CI Integration

Loctree is now integrated into:

1. **Pre-commit hooks** (`.pre-commit-config.yaml`):
   - `loctree-deadexports`: Checks for unused exports
   - `loctree-twins`: Checks for cross-language duplicates

2. **GitHub Actions** (`.github/workflows/loctree.yml`):
   - Runs on PRs and pushes to main branches
   - Uploads analysis report as artifact
   - Fails build if health score < 50

## Usage

```bash
# Full analysis
loct auto

# Check specific issues
loct twins --json
loct deadexports --json

# Get health score
cat .loctree/agent.json | jq '.summary.health_score'
```

---

**Created by M&K (c)2025 The LibraxisAI Team**

Co-Authored-By: Maciej Gad <void@div0.space>
Co-Authored-By: Klaudiusz <the1st@whoai.am>
