"""Policy wrapper that lets a local corridor-keeping model override goal policy."""
from collections import deque
from dataclasses import replace

from core.types import ACTION_TURN_LEFT, ACTION_TURN_RIGHT, FrameBatch, Prediction


class LccAssistPolicy:
    """Fuse a route policy with a local corridor-keeping policy.

    The route policy remains the default. The LCC policy only overrides when it
    sees a confident local correction. This keeps the upper-level route intent
    intact while giving the lower layer authority near walls, doors, and
    recoverable bad poses.
    """

    def __init__(
        self,
        route_policy,
        lcc_policy,
        lcc_threshold=0.55,
        override_frames=4,
        min_turn_bucket_conf=0.0,
    ):
        self.route_policy = route_policy
        self.lcc_policy = lcc_policy
        self.required_frames = max(route_policy.required_frames, lcc_policy.required_frames)
        self.lcc_threshold = lcc_threshold
        self.override_frames = override_frames
        self.min_turn_bucket_conf = min_turn_bucket_conf
        self.override_action = None
        self.override_left = 0
        self.recent_sources = deque(maxlen=32)

    def predict(self, batch: FrameBatch) -> Prediction:
        route_batch = FrameBatch(batch.frames[-self.route_policy.required_frames :])
        lcc_batch = FrameBatch(batch.frames[-self.lcc_policy.required_frames :])
        route_prediction = self.route_policy.predict(route_batch)
        lcc_prediction = self.lcc_policy.predict(lcc_batch)

        if self._should_override(lcc_prediction):
            self.override_action = lcc_prediction.action_id
            self.override_left = self.override_frames
            self.recent_sources.append("lcc")
            return replace(lcc_prediction, policy_source="lcc", policy_detail="override")

        if self.override_left > 0 and self.override_action == route_prediction.action_id:
            self.override_left -= 1
            self.recent_sources.append("route_hold")
            return replace(route_prediction, policy_source="route", policy_detail="lcc_hold_matched")

        self.override_left = max(0, self.override_left - 1)
        self.recent_sources.append("route")
        return replace(route_prediction, policy_source="route", policy_detail="default")

    def _should_override(self, prediction: Prediction) -> bool:
        if prediction.action_id not in (ACTION_TURN_LEFT, ACTION_TURN_RIGHT):
            return False
        if prediction.action_confidence < self.lcc_threshold:
            return False
        if prediction.mouse_confidence < self.min_turn_bucket_conf:
            return False
        return True
