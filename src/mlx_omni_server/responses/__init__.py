"""
OpenAI Responses API implementation for MLX Omni Server.

Provides /v1/responses endpoint compatible with OpenAI's Responses API,
with intelligent routing to local models and external providers.

Contributed by LibraxisAI - https://libraxis.ai
"""

from .router import router
from .schema import ResponseRequest, ResponseResponse

__all__ = ["ResponseRequest", "ResponseResponse", "router"]
