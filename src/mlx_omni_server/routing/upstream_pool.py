"""
Upstream pool with weighted round-robin selection.

Contributed by LibraxisAI - https://libraxis.ai
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass
class UpstreamTarget:
    """Single upstream target with weight."""

    url: str
    weight: int = 1
    name: str | None = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.url


class UpstreamPool:
    """
    Weighted round-robin selector for upstream targets.

    Usage:
        pool = UpstreamPool([
            UpstreamTarget("http://localhost:1234", weight=2),
            UpstreamTarget("http://localhost:11434", weight=1),
        ])

        target = pool.select()  # Returns 1234 twice as often as 11434
    """

    def __init__(self, targets: Iterable[UpstreamTarget]):
        self._targets: list[UpstreamTarget] = list(targets)

        # Expand by weights
        expanded: list[UpstreamTarget] = []
        for t in self._targets:
            expanded.extend([t] * max(1, t.weight))

        # Shuffle to avoid synchronization effects across workers
        random.shuffle(expanded)

        self._cycle: Iterator[UpstreamTarget] = itertools.cycle(
            expanded if expanded else self._targets
        )

    def select(self) -> UpstreamTarget:
        """O(1) amortized selection."""
        return next(self._cycle)

    @property
    def targets(self) -> list[UpstreamTarget]:
        """Get all configured targets."""
        return self._targets.copy()

    def __len__(self) -> int:
        return len(self._targets)
