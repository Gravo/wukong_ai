"""Facade for capture -> policy -> controller runtime loops."""
import csv
import time
from collections import deque
from pathlib import Path

from core.types import AgentStep, FrameBatch


class GameAgentFacade:
    """Small orchestration facade for inference and future DAgger loops."""

    def __init__(self, capture, policy, controller, conf_threshold=0.5, step_delay=0.05, gate=None):
        self.capture = capture
        self.policy = policy
        self.controller = controller
        self.conf_threshold = conf_threshold
        self.step_delay = step_delay
        self.gate = gate
        self.frames = deque(maxlen=policy.required_frames)

    def run_inference(self, duration=60, log_every=10, telemetry_path=None):
        start_time = time.time()
        executed = 0
        total_seen = 0
        steps = []
        telemetry_file = None
        telemetry_writer = None

        if telemetry_path:
            telemetry_path = Path(telemetry_path)
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            telemetry_file = telemetry_path.open("w", newline="", encoding="utf-8")
            telemetry_writer = csv.writer(telemetry_file)
            telemetry_writer.writerow(
                [
                    "elapsed",
                    "seen",
                    "executed",
                    "action_id",
                    "action_name",
                    "action_confidence",
                    "mouse_bucket",
                    "mouse_confidence",
                    "raw_mouse_dx",
                    "executed_step",
                    "reason",
                    "gate_mode",
                    "gate_threshold",
                ]
            )

        try:
            while time.time() - start_time < duration:
                frame = self.capture.grab()
                self.frames.append(frame)

                if len(self.frames) < self.policy.required_frames:
                    time.sleep(self.step_delay)
                    continue

                total_seen += 1
                prediction = self.policy.predict(FrameBatch(list(self.frames)))
                gate_mode = "none"
                gate_threshold = self.conf_threshold
                skip_reason = "low_confidence"
                if self.gate:
                    decision = self.gate.decide(prediction)
                    prediction = decision.prediction
                    gate_mode = decision.mode
                    gate_threshold = decision.threshold
                    skip_reason = decision.reason or "low_confidence"
                    should_execute = decision.execute
                else:
                    should_execute = prediction.action_confidence >= self.conf_threshold

                if not should_execute:
                    steps.append(AgentStep(executed, prediction, False, skip_reason))
                    if telemetry_writer:
                        telemetry_writer.writerow(
                            [
                                f"{time.time() - start_time:.3f}",
                                total_seen,
                                executed,
                                prediction.action_id,
                                prediction.action_name,
                                f"{prediction.action_confidence:.4f}",
                                prediction.mouse_bucket,
                                f"{prediction.mouse_confidence:.4f}",
                                prediction.raw_mouse_dx,
                                0,
                                skip_reason,
                                gate_mode,
                                f"{gate_threshold:.4f}",
                            ]
                        )
                    time.sleep(self.step_delay)
                    continue

                self.controller.execute(prediction)
                executed += 1
                steps.append(AgentStep(executed, prediction, True))
                if telemetry_writer:
                    telemetry_writer.writerow(
                        [
                            f"{time.time() - start_time:.3f}",
                            total_seen,
                            executed,
                            prediction.action_id,
                            prediction.action_name,
                            f"{prediction.action_confidence:.4f}",
                            prediction.mouse_bucket,
                            f"{prediction.mouse_confidence:.4f}",
                            prediction.raw_mouse_dx,
                            1,
                            "",
                            gate_mode,
                            f"{gate_threshold:.4f}",
                        ]
                    )

                if log_every and executed % log_every == 0:
                    print(
                        f"[agent] step={executed} action={prediction.action_name} "
                        f"conf={prediction.action_confidence:.2f} "
                        f"bucket={prediction.mouse_bucket} "
                        f"mouse_conf={prediction.mouse_confidence:.2f}",
                        flush=True,
                    )

                time.sleep(self.step_delay)
        except KeyboardInterrupt:
            print("\n[agent] interrupted", flush=True)
        finally:
            self.controller.release_all()
            close = getattr(self.controller, "close", None)
            if close:
                close()
            self.capture.close()
            if telemetry_file:
                telemetry_file.close()

        return steps
