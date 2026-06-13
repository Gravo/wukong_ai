"""DIP ports: high-level agent code depends on these abstractions."""
from typing import Protocol

import numpy as np

from core.types import FrameBatch, Prediction


class CapturePort(Protocol):
    """Frame source abstraction."""

    def grab(self) -> np.ndarray:
        """Return one BGR frame."""
        ...

    def close(self) -> None:
        """Release capture resources."""
        ...


class PolicyPort(Protocol):
    """Policy abstraction for any model version."""

    @property
    def required_frames(self) -> int:
        """Number of frames required for one prediction."""
        ...

    def predict(self, batch: FrameBatch) -> Prediction:
        """Return an action prediction for the current frame window."""
        ...


class ControllerPort(Protocol):
    """Game control abstraction."""

    def execute(self, prediction: Prediction) -> None:
        """Execute one model prediction in game."""
        ...

    def release_all(self) -> None:
        """Release held inputs."""
        ...


class RunRegistryPort(Protocol):
    """Persistence abstraction for runtime metadata."""

    def create_run(self, config, metadata: dict | None = None) -> int:
        """Create one run record and return its id."""
        ...

    def finish_run(self, run_id: int, steps, metrics: dict | None = None) -> None:
        """Persist final metrics for a run."""
        ...

    def close(self) -> None:
        """Release registry resources."""
        ...
