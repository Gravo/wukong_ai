"""PyAutoGUI capture adapter used by the legacy v5.5 inference script."""
import cv2
import numpy as np
import pyautogui


class PyAutoGuiCapture:
    """Simple full-screen screenshot capture returning BGR frames."""

    def grab(self) -> np.ndarray:
        frame = pyautogui.screenshot()
        frame = np.array(frame)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        pass

