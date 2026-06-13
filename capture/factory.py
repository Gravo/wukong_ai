"""Capture adapter factory."""


def build_capture(name: str):
    if name == "screen":
        from capture.screen_capture_adapter import ScreenCaptureAdapter

        return ScreenCaptureAdapter()
    if name == "pyautogui":
        from capture.pyautogui_capture import PyAutoGuiCapture

        return PyAutoGuiCapture()
    raise ValueError(f"Unknown capture backend: {name}")
