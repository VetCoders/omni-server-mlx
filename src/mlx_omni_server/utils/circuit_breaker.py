"""
Circuit Breaker implementation for upstream service protection.

Implements the classic three-state pattern:
- CLOSED: Normal operation, requests pass through
- OPEN: Service failing, requests rejected immediately
- HALF_OPEN: Testing recovery, limited requests allowed

Contributed by LibraxisAI - https://libraxis.ai
"""

from __future__ import annotations

import asyncio
import threading
import time
from enum import Enum
from typing import Any, Callable, TypeVar

from ..utils.logger import logger

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open and rejecting requests."""

    def __init__(self, name: str, remaining_timeout: float):
        self.name = name
        self.remaining_timeout = remaining_timeout
        super().__init__(
            f"Circuit breaker '{name}' is OPEN. "
            f"Retry after {remaining_timeout:.1f}s"
        )


class CircuitBreaker:
    """
    Async-safe circuit breaker for protecting upstream services.

    Usage:
        breaker = CircuitBreaker(name="ollama", failure_threshold=5)

        # Check before call
        if breaker.is_open():
            raise CircuitBreakerOpen(...)

        try:
            result = await call_upstream()
            breaker.record_success()
        except Exception:
            breaker.record_failure()
            raise

    Or use the context manager:
        async with breaker.protect():
            result = await call_upstream()
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        timeout: int = 60,
        success_threshold: int = 2,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for logging
            failure_threshold: Failures before opening circuit
            timeout: Seconds before attempting recovery
            success_threshold: Successes in half-open to close circuit
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def failures(self) -> int:
        """Current failure count."""
        return self._failure_count

    def is_open(self) -> bool:
        """
        Check if circuit is open (blocking requests).

        Also handles auto-recovery transition to HALF_OPEN.
        """
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.timeout:
                    # Auto-recover to half-open
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(
                        f"Circuit breaker '{self.name}' entering HALF_OPEN "
                        f"after {elapsed:.1f}s timeout"
                    )
                    return False
            return True
        return False

    def remaining_timeout(self) -> float:
        """Seconds until recovery attempt (0 if not open)."""
        if self._state != CircuitState.OPEN or self._last_failure_time is None:
            return 0.0
        elapsed = time.time() - self._last_failure_time
        return max(0.0, self.timeout - elapsed)

    def record_success(self) -> None:
        """
        Record a successful request.

        In HALF_OPEN state, may transition to CLOSED after enough successes.
        In CLOSED state, resets failure count.
        """
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info(
                    f"Circuit breaker '{self.name}' CLOSED "
                    f"after {self.success_threshold} successful requests"
                )
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """
        Record a failed request.

        May trigger OPEN state if threshold reached.
        In HALF_OPEN state, immediately re-opens circuit.
        """
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Immediate re-open on any failure during recovery
            self._state = CircuitState.OPEN
            self._success_count = 0
            logger.warning(
                f"Circuit breaker '{self.name}' re-OPENED "
                f"after failure during recovery"
            )
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker '{self.name}' OPENED "
                f"after {self._failure_count} failures"
            )

    def reset(self) -> None:
        """Force reset to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")

    def get_status(self) -> dict[str, Any]:
        """Get current status for health checks."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failures": self._failure_count,
            "successes": self._success_count,
            "remaining_timeout": self.remaining_timeout(),
        }

    async def call(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async or sync callable to execute
            *args, **kwargs: Arguments to pass to func

        Returns:
            Result of func

        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Any exception from func (after recording failure)
        """
        if self.is_open():
            raise CircuitBreakerOpen(self.name, self.remaining_timeout())

        try:
            result = func(*args, **kwargs)
            # Handle async functions
            if asyncio.iscoroutine(result):
                result = await result
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise


# Global registry of circuit breakers by name
_circuit_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: int = 60,
    success_threshold: int = 2,
) -> CircuitBreaker:
    """
    Get or create a named circuit breaker.

    Thread-safe factory for shared circuit breakers using double-checked locking.
    """
    if name not in _circuit_breakers:
        with _registry_lock:
            # Double-check after acquiring lock to prevent race condition
            if name not in _circuit_breakers:
                _circuit_breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    timeout=timeout,
                    success_threshold=success_threshold,
                )
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> dict[str, dict[str, Any]]:
    """Get status of all registered circuit breakers."""
    return {name: cb.get_status() for name, cb in _circuit_breakers.items()}
