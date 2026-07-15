"""Helpers for severe class imbalance (issue data/#3)."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


def class_weights(labels: Iterable[str | int]) -> dict:
    """Inverse-frequency weights for loss rebalancing."""
    counts = Counter(labels)
    if not counts:
        return {}
    total = sum(counts.values())
    n_classes = len(counts)
    return {c: total / (n_classes * n) for c, n in counts.items()}


def undersample_indices(labels: list, max_per_class: int | None = None) -> list[int]:
    """Return indices keeping at most max_per_class per label (default: median count)."""
    counts = Counter(labels)
    if not counts:
        return []
    if max_per_class is None:
        ordered = sorted(counts.values())
        max_per_class = ordered[len(ordered) // 2]
    seen: Counter = Counter()
    out: list[int] = []
    for i, lab in enumerate(labels):
        if seen[lab] < max_per_class:
            out.append(i)
            seen[lab] += 1
    return out
