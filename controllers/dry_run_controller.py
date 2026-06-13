"""Controller that logs predictions without touching the game."""
from core.types import Prediction


class DryRunController:
    """Useful for validating policy and orchestration without sending inputs."""

    def __init__(self, log_every=1):
        self.log_every = log_every
        self.count = 0

    def execute(self, prediction: Prediction) -> None:
        self.count += 1
        if self.log_every and self.count % self.log_every == 0:
            print(
                f"[dry-run] action={prediction.action_name} "
                f"conf={prediction.action_confidence:.2f} "
                f"bucket={prediction.mouse_bucket} "
                f"dx={prediction.raw_mouse_dx}",
                flush=True,
            )

    def release_all(self) -> None:
        pass

