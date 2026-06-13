"""Keyboard/mouse controller backed by pydirectinput."""
import time

import pydirectinput as pdi

from core.types import ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT, Prediction


class PyDirectController:
    """Legacy-compatible controller for keyboard and mouse movement."""

    def __init__(self, mouse_interval=0.05, smoothing=0.7, micro_step_px=10):
        self.mouse_interval = mouse_interval
        self.smoothing = smoothing
        self.micro_step_px = micro_step_px
        self.w_pressed = False
        self.last_mouse_time = 0.0
        self.last_dx = 0.0

    def execute(self, prediction: Prediction) -> None:
        if prediction.action_id == ACTION_FORWARD:
            if not self.w_pressed:
                pdi.keyDown("w")
                self.w_pressed = True
        else:
            if self.w_pressed:
                pdi.keyUp("w")
                self.w_pressed = False

            if prediction.action_id == ACTION_TURN_LEFT:
                pdi.keyDown("a")
                time.sleep(0.05)
                pdi.keyUp("a")
            elif prediction.action_id == ACTION_TURN_RIGHT:
                pdi.keyDown("d")
                time.sleep(0.05)
                pdi.keyUp("d")

        self._move_mouse(prediction.raw_mouse_dx)

    def _move_mouse(self, dx: int) -> None:
        current_time = time.time()
        if current_time - self.last_mouse_time < self.mouse_interval:
            return

        smoothed_dx = self.smoothing * dx + (1.0 - self.smoothing) * self.last_dx
        self.last_dx = smoothed_dx

        steps = max(1, int(abs(smoothed_dx) / self.micro_step_px))
        micro_dx = smoothed_dx / steps
        for _ in range(steps):
            pdi.moveRel(int(micro_dx), 0, relative=True)
            time.sleep(0.01)

        self.last_mouse_time = current_time

    def release_all(self) -> None:
        try:
            pdi.keyUp("w")
            pdi.keyUp("a")
            pdi.keyUp("d")
        finally:
            self.w_pressed = False

