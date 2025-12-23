"""
Tests for circuit breaker implementation.

Contributed by LibraxisAI - https://libraxis.ai
"""

import time

import pytest

from mlx_omni_server.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
    get_all_circuit_breakers,
    get_circuit_breaker,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Circuit breaker should start in CLOSED state."""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failures == 0
        assert not breaker.is_open()

    def test_record_success_resets_failure_count(self):
        """Recording success should reset failure count."""
        breaker = CircuitBreaker(name="test", failure_threshold=5)

        # Simulate some failures
        for _ in range(3):
            breaker.record_failure()

        assert breaker.failures == 3

        # Success should reset
        breaker.record_success()
        assert breaker.failures == 0

    def test_opens_after_threshold_failures(self):
        """Circuit should OPEN after reaching failure threshold."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        for i in range(3):
            breaker.record_failure()
            if i < 2:
                assert breaker.state == CircuitState.CLOSED

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open()

    def test_rejects_when_open(self):
        """Circuit should reject when OPEN."""
        breaker = CircuitBreaker(name="test", failure_threshold=1)
        breaker.record_failure()

        assert breaker.is_open()

        with pytest.raises(CircuitBreakerOpen) as exc_info:
            raise CircuitBreakerOpen(breaker.name, breaker.remaining_timeout())

        assert "test" in str(exc_info.value)

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit should transition to HALF_OPEN after timeout."""
        breaker = CircuitBreaker(name="test", failure_threshold=1, timeout=1)
        breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open()

        # Wait for timeout
        time.sleep(1.1)

        # is_open() should trigger transition
        assert not breaker.is_open()
        assert breaker.state == CircuitState.HALF_OPEN

    def test_closes_after_success_threshold_in_half_open(self):
        """Circuit should CLOSE after success threshold in HALF_OPEN."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=1,
            timeout=0,  # Immediate recovery
            success_threshold=2,
        )

        # Open circuit
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Trigger half-open
        breaker.is_open()
        assert breaker.state == CircuitState.HALF_OPEN

        # First success
        breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN

        # Second success should close
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        """Circuit should reopen immediately on failure in HALF_OPEN."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=1,
            timeout=0,
            success_threshold=2,
        )

        # Open and transition to half-open
        breaker.record_failure()
        breaker.is_open()
        assert breaker.state == CircuitState.HALF_OPEN

        # Failure in half-open should reopen
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_reset_clears_state(self):
        """Manual reset should clear all state."""
        breaker = CircuitBreaker(name="test", failure_threshold=1)
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failures == 0

    def test_get_status_returns_info(self):
        """get_status should return complete status info."""
        breaker = CircuitBreaker(name="test-status")
        status = breaker.get_status()

        assert status["name"] == "test-status"
        assert status["state"] == "closed"
        assert status["failures"] == 0

    @pytest.mark.asyncio
    async def test_async_call_records_success(self):
        """Async call should record success on completion."""
        breaker = CircuitBreaker(name="test")

        async def async_func():
            return "result"

        result = await breaker.call(async_func)
        assert result == "result"
        assert breaker.failures == 0

    @pytest.mark.asyncio
    async def test_async_call_records_failure(self):
        """Async call should record failure on exception."""
        breaker = CircuitBreaker(name="test")

        async def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await breaker.call(failing_func)

        assert breaker.failures == 1


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry functions."""

    def test_get_circuit_breaker_creates_new(self):
        """get_circuit_breaker should create new breaker if not exists."""
        breaker = get_circuit_breaker("registry-test-1")
        assert breaker.name == "registry-test-1"

    def test_get_circuit_breaker_returns_same(self):
        """get_circuit_breaker should return same instance for same name."""
        breaker1 = get_circuit_breaker("registry-test-2")
        breaker2 = get_circuit_breaker("registry-test-2")
        assert breaker1 is breaker2

    def test_get_all_circuit_breakers(self):
        """get_all_circuit_breakers should return all registered breakers."""
        get_circuit_breaker("registry-test-3")
        all_breakers = get_all_circuit_breakers()
        assert "registry-test-3" in all_breakers
