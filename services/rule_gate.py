"""BLUE-inspired lightweight runtime gate for policy execution."""
from dataclasses import dataclass, replace
from collections import deque

from core.types import (
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    ACTION_NAMES,
    Prediction,
)


@dataclass(frozen=True)
class GateDecision:
    prediction: Prediction
    execute: bool
    threshold: float
    mode: str
    reason: str


class RuleGate:
    """A small hand-written gate that opens recovery behavior on weak turns.

    This is the first baseline for a BLUE-style gate. It does not change the
    model; it decides whether to trust weak correction signals and makes turn
    predictions executable when the mouse bucket collapses to straight.
    """

    def __init__(
        self,
        base_threshold=0.5,
        recovery_threshold=0.35,
        forward_threshold=0.35,
        turn_dx=100,
        history=20,
        bucket3_patience=12,
        turn_hold_frames=4,
    ):
        self.base_threshold = base_threshold
        self.recovery_threshold = recovery_threshold
        self.forward_threshold = forward_threshold
        self.turn_dx = turn_dx
        self.recent = deque(maxlen=history)
        self.bucket3_patience = bucket3_patience
        self.turn_hold_frames = turn_hold_frames
        self.held_turn = None
        self.held_turn_left = 0

    def decide(self, prediction: Prediction) -> GateDecision:
        is_turn = prediction.action_id in (ACTION_TURN_LEFT, ACTION_TURN_RIGHT)
        bucket3_streak = self._bucket3_streak()
        mode = "normal"
        threshold = self.base_threshold
        reason = ""
        gated_prediction = prediction

        if is_turn and self._should_hold_turn(prediction):
            mode = "held_turn"
            threshold = self.recovery_threshold
            reason = "turn_hysteresis"
            gated_prediction = self._ensure_turn_dx(self._with_action(prediction, self.held_turn))
        elif is_turn and prediction.action_confidence >= self.recovery_threshold:
            mode = "recovery_turn"
            threshold = self.recovery_threshold
            reason = "weak_turn_allowed"
            gated_prediction = self._ensure_turn_dx(prediction)
            self.held_turn = prediction.action_id
            self.held_turn_left = self.turn_hold_frames
        elif prediction.action_confidence >= self.forward_threshold:
            mode = "low_conf_forward"
            threshold = self.forward_threshold
            reason = "weak_forward_allowed"
        elif bucket3_streak >= self.bucket3_patience and prediction.action_confidence >= self.recovery_threshold:
            mode = "low_conf_patrol"
            threshold = self.recovery_threshold
            reason = "bucket3_streak"

        execute = gated_prediction.action_confidence >= threshold
        if not execute and not reason:
            reason = "low_confidence"

        self.recent.append(gated_prediction)
        if execute and gated_prediction.action_id not in (ACTION_TURN_LEFT, ACTION_TURN_RIGHT):
            self.held_turn = None
            self.held_turn_left = 0
        elif self.held_turn_left > 0:
            self.held_turn_left -= 1
        return GateDecision(
            prediction=gated_prediction,
            execute=execute,
            threshold=threshold,
            mode=mode,
            reason="" if execute else reason,
        )

    def _bucket3_streak(self) -> int:
        streak = 0
        for item in reversed(self.recent):
            if item.mouse_bucket == 3:
                streak += 1
            else:
                break
        return streak

    def _ensure_turn_dx(self, prediction: Prediction) -> Prediction:
        if prediction.raw_mouse_dx != 0:
            return prediction
        if prediction.action_id == ACTION_TURN_LEFT:
            bucket = 1 if self.turn_dx >= 100 else 2
            raw_dx = -abs(self.turn_dx)
        elif prediction.action_id == ACTION_TURN_RIGHT:
            bucket = 5 if self.turn_dx >= 100 else 4
            raw_dx = abs(self.turn_dx)
        else:
            return prediction
        return replace(
            prediction,
            mouse_bucket=bucket,
            action_name=ACTION_NAMES.get(prediction.action_id, prediction.action_name),
            raw_mouse_dx=raw_dx,
        )

    def _should_hold_turn(self, prediction: Prediction) -> bool:
        if self.held_turn_left <= 0 or self.held_turn is None:
            return False
        if prediction.action_id == self.held_turn:
            return False
        return prediction.action_confidence < self.base_threshold

    def _with_action(self, prediction: Prediction, action_id: int) -> Prediction:
        return replace(
            prediction,
            action_id=action_id,
            action_name=ACTION_NAMES.get(action_id, prediction.action_name),
        )
