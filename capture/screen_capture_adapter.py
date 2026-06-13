"""Adapter around env.screen_capture.ScreenCapture."""
from env.screen_capture import ScreenCapture


class ScreenCaptureAdapter:
    """Use the existing dxcam/mss/win32 capture implementation as a port."""

    def __init__(self, backend=None, region=None, target_fps=None):
        self._capture = ScreenCapture(
            backend=backend,
            region=region,
            target_fps=target_fps,
        )

    def grab(self):
        return self._capture.grab()

    def close(self) -> None:
        self._capture.release()

