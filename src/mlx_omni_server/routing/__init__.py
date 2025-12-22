"""
Multi-provider routing module for MLX Omni Server.

Provides intelligent routing across multiple LLM providers with:
- Round-robin load balancing
- Fallback chains
- Provider-specific authentication
- Health checking

Contributed by LibraxisAI - https://libraxis.ai
"""

from .multi_provider import (
    MultiProviderRouter,
    Provider,
    ProviderType,
    get_default_router,
)
from .upstream_pool import UpstreamPool, UpstreamTarget

__all__ = [
    "MultiProviderRouter",
    "Provider",
    "ProviderType",
    "UpstreamPool",
    "UpstreamTarget",
    "get_default_router",
]
