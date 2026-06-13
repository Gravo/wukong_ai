"""Runtime configuration for agent applications."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AgentRuntimeConfig:
    """Configuration that wires capture, policy, and controller adapters."""

    model_path: Path
    goal_id: int = 0
    duration: int = 60
    conf_threshold: float = 0.5
    device: str = "cuda:0"
    capture: str = "screen"
    controller: str = "vigem"
    policy: str = "v55"
    step_delay: float = 0.05
    log_every: int = 10
    run_name: Optional[str] = None
    registry_path: Optional[Path] = None
    telemetry_path: Optional[Path] = None
    always_forward: bool = True
    gate: str = "none"
    recovery_threshold: float = 0.35
    gate_forward_threshold: float = 0.35
    gate_turn_dx: int = 100
    gate_turn_hold_frames: int = 4
