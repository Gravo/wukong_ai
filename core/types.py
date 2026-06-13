"""Shared data types for capture, policy, and control layers."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


ACTION_FORWARD = 0
ACTION_TURN_LEFT = 1
ACTION_TURN_RIGHT = 2

ACTION_NAMES = {
    ACTION_FORWARD: "forward",
    ACTION_TURN_LEFT: "turn_left",
    ACTION_TURN_RIGHT: "turn_right",
}

MOUSE_BUCKET_TO_DX = {
    0: -300,
    1: -150,
    2: -50,
    3: 0,
    4: 50,
    5: 150,
    6: 300,
}


@dataclass(frozen=True)
class FrameBatch:
    """A short temporal window of BGR frames."""

    frames: list[np.ndarray]


@dataclass(frozen=True)
class Prediction:
    """Model output after policy-level post-processing."""

    action_id: int
    action_confidence: float
    mouse_bucket: int
    mouse_confidence: float
    action_name: str
    raw_mouse_dx: int


@dataclass(frozen=True)
class AgentStep:
    """One completed agent step, useful for logging and DAgger later."""

    step_index: int
    prediction: Optional[Prediction]
    executed: bool
    reason: str = ""

