"""
Multi-provider routing with fallback support.

Routes requests to multiple LLM providers (MLX local, Ollama, LM Studio,
OpenAI, etc.) with intelligent fallback and load balancing.

Contributed by LibraxisAI - https://libraxis.ai
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import httpx

from ..utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    get_circuit_breaker,
)
from ..utils.logger import logger

T = TypeVar("T")


class ProviderType(Enum):
    """Supported provider types."""

    MLX = "mlx"           # Local MLX models
    OLLAMA = "ollama"     # Ollama server
    LMSTUDIO = "lmstudio"  # LM Studio
    OPENAI = "openai"     # OpenAI API
    ANTHROPIC = "anthropic"  # Anthropic API
    CUSTOM = "custom"     # Custom OpenAI-compatible endpoint


@dataclass
class Provider:
    """Provider configuration."""

    name: str
    base_url: str
    provider_type: ProviderType
    priority: int = 10  # Lower = higher priority (1 = try first)
    auth_type: str = "none"  # "none", "bearer", "header"
    api_key_env: str | None = None  # Env var name for API key
    timeout: int = 120
    enabled: bool = True
    weight: int = 1  # For weighted round-robin

    # Runtime state
    _api_key: str | None = field(default=None, repr=False)
    _circuit_breaker: CircuitBreaker | None = field(default=None, repr=False)

    def __post_init__(self):
        # Load API key from environment
        if self.api_key_env:
            self._api_key = os.environ.get(self.api_key_env)

        # Create circuit breaker
        self._circuit_breaker = get_circuit_breaker(
            name=f"provider-{self.name}",
            failure_threshold=5,
            timeout=60,
            success_threshold=2,
        )

    @property
    def api_key(self) -> str | None:
        """Get API key (loaded from env)."""
        return self._api_key

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get circuit breaker for this provider."""
        if self._circuit_breaker is None:
            self._circuit_breaker = get_circuit_breaker(f"provider-{self.name}")
        return self._circuit_breaker

    def get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for this provider."""
        if self.auth_type == "none" or not self._api_key:
            return {}

        if self.auth_type == "bearer":
            return {"Authorization": f"Bearer {self._api_key}"}
        elif self.auth_type == "header":
            # X-API-Key style
            return {"X-API-Key": self._api_key}

        return {}

    def is_available(self) -> bool:
        """Check if provider is enabled and circuit is not open."""
        if not self.enabled:
            return False
        return not self.circuit_breaker.is_open()


class AllProvidersFailedError(Exception):
    """Raised when all providers in the chain have failed."""

    def __init__(self, errors: list[tuple[str, Exception]]):
        self.errors = errors
        error_summary = "; ".join(f"{name}: {e}" for name, e in errors)
        super().__init__(f"All providers failed: {error_summary}")


class MultiProviderRouter:
    """
    Routes requests across multiple LLM providers with fallback.

    Features:
    - Round-robin load balancing within same priority tier
    - Automatic fallback to lower priority providers on failure
    - Circuit breaker per provider
    - Configurable retry with exponential backoff

    Usage:
        router = MultiProviderRouter()
        router.add_provider(Provider(
            name="ollama",
            base_url="http://localhost:11434",
            provider_type=ProviderType.OLLAMA,
            priority=1,
        ))

        result = await router.call(
            endpoint="/v1/chat/completions",
            payload={"model": "...", "messages": [...]},
        )
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_multiplier: float = 2.0,
    ):
        self.providers: list[Provider] = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_multiplier = retry_multiplier

        self._rr_index = 0
        self._lock = asyncio.Lock()

    def add_provider(self, provider: Provider) -> None:
        """Add a provider to the routing chain."""
        self.providers.append(provider)
        # Sort by priority (lower = higher priority)
        self.providers.sort(key=lambda p: p.priority)
        logger.info(
            f"Added provider '{provider.name}' "
            f"({provider.provider_type.value}) at priority {provider.priority}"
        )

    def add_from_env(self) -> None:
        """
        Auto-discover providers from environment variables.

        Looks for:
        - LLM_BASE_URL / LLM_ALT_BASE_URL
        - LLM_BASE_URLS (JSON array or CSV)
        - OLLAMA_API_URL
        - Provider-specific API keys (OPENAI_API_KEY, etc.)
        """
        # Primary local provider
        llm_base_url = os.environ.get("LLM_BASE_URL")
        if llm_base_url:
            self.add_provider(Provider(
                name="primary",
                base_url=llm_base_url.rstrip("/"),
                provider_type=self._detect_provider_type(llm_base_url),
                priority=1,
            ))

        # Secondary local provider
        llm_alt_url = os.environ.get("LLM_ALT_BASE_URL")
        if llm_alt_url:
            self.add_provider(Provider(
                name="secondary",
                base_url=llm_alt_url.rstrip("/"),
                provider_type=self._detect_provider_type(llm_alt_url),
                priority=2,
            ))

        # Additional URLs from list
        llm_base_urls = self._parse_url_list(os.environ.get("LLM_BASE_URLS", ""))
        for i, url in enumerate(llm_base_urls):
            if url not in [llm_base_url, llm_alt_url]:
                self.add_provider(Provider(
                    name=f"upstream-{i+1}",
                    base_url=url.rstrip("/"),
                    provider_type=self._detect_provider_type(url),
                    priority=3 + i,
                ))

        # Ollama (if configured separately)
        ollama_url = os.environ.get("OLLAMA_API_URL")
        if ollama_url and ollama_url not in [llm_base_url, llm_alt_url]:
            self.add_provider(Provider(
                name="ollama",
                base_url=ollama_url.rstrip("/"),
                provider_type=ProviderType.OLLAMA,
                priority=5,
            ))

        # Cloud providers (only if API keys present)
        if os.environ.get("OPENAI_API_KEY"):
            self.add_provider(Provider(
                name="openai",
                base_url="https://api.openai.com/v1",
                provider_type=ProviderType.OPENAI,
                priority=10,
                auth_type="bearer",
                api_key_env="OPENAI_API_KEY",
            ))

    def _detect_provider_type(self, url: str) -> ProviderType:
        """Detect provider type from URL pattern."""
        url_lower = url.lower()
        if "11434" in url_lower or "ollama" in url_lower:
            return ProviderType.OLLAMA
        elif "1234" in url_lower or "lmstudio" in url_lower:
            return ProviderType.LMSTUDIO
        elif "openai.com" in url_lower:
            return ProviderType.OPENAI
        elif "anthropic.com" in url_lower:
            return ProviderType.ANTHROPIC
        return ProviderType.CUSTOM

    def _parse_url_list(self, raw: str) -> list[str]:
        """Parse URL list from JSON array or comma-separated string."""
        if not raw or not raw.strip():
            return []

        raw = raw.strip()

        # Try JSON first
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(u).strip() for u in parsed if u]
        except json.JSONDecodeError:
            pass

        # Fallback to comma-separated
        return [part.strip() for part in raw.split(",") if part.strip()]

    def get_available_providers(self) -> list[Provider]:
        """Get list of currently available providers (enabled + circuit closed)."""
        return [p for p in self.providers if p.is_available()]

    async def select_provider(self) -> Provider | None:
        """
        Select next provider using round-robin within priority tiers.

        Returns highest-priority available provider, rotating among
        same-priority providers.
        """
        available = self.get_available_providers()
        if not available:
            return None

        # Group by priority
        by_priority: dict[int, list[Provider]] = {}
        for p in available:
            by_priority.setdefault(p.priority, []).append(p)

        # Get highest priority tier (lowest number)
        min_priority = min(by_priority.keys())
        tier = by_priority[min_priority]

        if len(tier) == 1:
            return tier[0]

        # Round-robin within tier
        async with self._lock:
            idx = self._rr_index % len(tier)
            self._rr_index += 1
            return tier[idx]

    async def call(
        self,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | httpx.Response:
        """
        Call endpoint with automatic provider fallback.

        Args:
            endpoint: API endpoint path (e.g., "/v1/chat/completions")
            payload: Request body
            headers: Additional headers (merged with provider auth)
            stream: If True, returns raw httpx.Response for streaming

        Returns:
            JSON response dict (or httpx.Response if stream=True)

        Raises:
            AllProvidersFailedError: If all providers fail
        """
        errors: list[tuple[str, Exception]] = []
        request_headers = dict(headers) if headers else {}
        request_headers.setdefault("Content-Type", "application/json")

        for provider in self.providers:
            if not provider.is_available():
                logger.debug(f"Skipping unavailable provider: {provider.name}")
                continue

            # Merge provider auth headers
            full_headers = {**request_headers, **provider.get_auth_headers()}

            # Build full URL
            url = self._build_url(provider, endpoint)

            try:
                result = await self._call_with_retry(
                    provider=provider,
                    url=url,
                    payload=payload,
                    headers=full_headers,
                    stream=stream,
                )
                provider.circuit_breaker.record_success()
                logger.debug(f"Request succeeded via {provider.name}")
                return result

            except CircuitBreakerOpen as e:
                logger.debug(f"Circuit open for {provider.name}: {e}")
                errors.append((provider.name, e))
                continue

            except Exception as e:
                provider.circuit_breaker.record_failure()
                logger.warning(f"Provider {provider.name} failed: {e}")
                errors.append((provider.name, e))
                continue

        raise AllProvidersFailedError(errors)

    def _build_url(self, provider: Provider, endpoint: str) -> str:
        """Build full URL for provider and endpoint."""
        base = provider.base_url.rstrip("/")
        endpoint = endpoint.lstrip("/")

        # Handle provider-specific URL patterns
        if provider.provider_type == ProviderType.OLLAMA:
            # Ollama uses /api/chat for chat completions
            if "chat/completions" in endpoint:
                return f"{base}/api/chat"
            elif "models" in endpoint:
                return f"{base}/api/tags"

        # OpenAI-compatible endpoints
        if not base.endswith("/v1"):
            return f"{base}/v1/{endpoint}"
        return f"{base}/{endpoint}"

    async def _call_with_retry(
        self,
        provider: Provider,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        stream: bool,
    ) -> dict[str, Any] | httpx.Response:
        """Make request with exponential backoff retry."""
        delay = self.retry_delay

        for attempt in range(self.max_retries + 1):
            try:
                timeout = httpx.Timeout(provider.timeout, connect=10.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    if stream:
                        # Return response for streaming
                        response = await client.post(
                            url,
                            json=payload,
                            headers=headers,
                        )
                        response.raise_for_status()
                        return response
                    else:
                        response = await client.post(
                            url,
                            json=payload,
                            headers=headers,
                        )
                        response.raise_for_status()
                        return response.json()

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt == self.max_retries:
                    raise
                # Exponential backoff with jitter
                jitter = delay * (0.5 + random.random())
                logger.debug(
                    f"Retry {attempt + 1}/{self.max_retries} "
                    f"for {provider.name} after {jitter:.2f}s: {e}"
                )
                await asyncio.sleep(jitter)
                delay *= self.retry_multiplier

            except httpx.HTTPStatusError as e:
                # Don't retry client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    raise
                if attempt == self.max_retries:
                    raise
                # Retry server errors (5xx)
                jitter = delay * (0.5 + random.random())
                await asyncio.sleep(jitter)
                delay *= self.retry_multiplier

        # Should not reach here
        raise RuntimeError("Retry loop exhausted without raising")

    def get_status(self) -> dict[str, Any]:
        """Get router status for health checks."""
        return {
            "providers": [
                {
                    "name": p.name,
                    "type": p.provider_type.value,
                    "base_url": p.base_url,
                    "priority": p.priority,
                    "enabled": p.enabled,
                    "available": p.is_available(),
                    "circuit_breaker": p.circuit_breaker.get_status(),
                }
                for p in self.providers
            ],
            "available_count": len(self.get_available_providers()),
            "total_count": len(self.providers),
        }


# Global default router instance
_default_router: MultiProviderRouter | None = None


def get_default_router() -> MultiProviderRouter:
    """Get or create the default router with env-based configuration."""
    global _default_router
    if _default_router is None:
        _default_router = MultiProviderRouter()
        _default_router.add_from_env()
    return _default_router
