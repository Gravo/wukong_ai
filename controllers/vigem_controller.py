"""ViGEm virtual gamepad controller for Raw Input-resistant games."""
import time

from core.types import ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT, Prediction


class ViGEmController:
    """Map policy predictions to an Xbox 360 virtual controller."""

    BUCKET_TO_STICK = {
        -300: -0.85,
        -150: -0.55,
        -50: -0.25,
        0: 0.0,
        50: 0.25,
        150: 0.55,
        300: 0.85,
    }

    def __init__(self, turn_scale=110.0, forward_value=-32768, always_forward=True):
        self.backend = None
        self.vgamepad = None
        self.pad = None
        self.vigemclient = None
        self.client = None
        self.gamepad = None
        self.turn_scale = turn_scale
        self.forward_value = forward_value
        self.always_forward = always_forward

        try:
            self._init_vgamepad()
            return
        except ImportError:
            pass

        try:
            self._init_vigemclient()
            return
        except ImportError as exc:
            raise RuntimeError(
                "A ViGEm Python package is required. Install ViGEmBus first, then run "
                "`C:\\Python\\python.exe -m pip install vgamepad` "
                "(preferred) or `C:\\Python\\python.exe -m pip install vigemclient`."
            ) from exc

    def _init_vgamepad(self) -> None:
        import vgamepad

        self.backend = "vgamepad"
        self.vgamepad = vgamepad
        self.pad = vgamepad.VX360Gamepad()
        # Prime the virtual pad so the game notices controller mode.
        self.pad.right_joystick_float(0.1, 0.0)
        self.pad.update()
        time.sleep(0.05)
        self.pad.right_joystick_float(0.0, 0.0)
        self.pad.update()

    def _init_vigemclient(self) -> None:
        import vigemclient

        self.backend = "vigemclient"
        self.vigemclient = vigemclient
        self.client = vigemclient.VigemClient()
        self.client.connect()
        self.gamepad = vigemclient.VigemXbox360Controller(self.client)
        self.gamepad.register()

    def execute(self, prediction: Prediction) -> None:
        if self.backend == "vgamepad":
            self._execute_vgamepad(prediction)
        else:
            self._execute_vigemclient(prediction)

    def _execute_vgamepad(self, prediction: Prediction) -> None:
        left_y = 0.0
        if self.always_forward or prediction.action_id == ACTION_FORWARD:
            left_y = 1.0

        right_x = self.BUCKET_TO_STICK.get(prediction.raw_mouse_dx, 0.0)
        if prediction.action_id == ACTION_TURN_LEFT and right_x >= 0:
            right_x = -0.45
        elif prediction.action_id == ACTION_TURN_RIGHT and right_x <= 0:
            right_x = 0.45

        self.pad.left_joystick_float(0.0, left_y)
        self.pad.right_joystick_float(right_x, 0.0)
        self.pad.update()
        time.sleep(0.02)

    def _execute_vigemclient(self, prediction: Prediction) -> None:
        left_y = 0
        if self.always_forward or prediction.action_id == ACTION_FORWARD:
            left_y = self.forward_value

        right_x = int(max(-32768, min(32767, prediction.raw_mouse_dx * self.turn_scale)))
        if prediction.action_id == ACTION_TURN_LEFT and right_x >= 0:
            right_x = -16000
        elif prediction.action_id == ACTION_TURN_RIGHT and right_x <= 0:
            right_x = 16000

        self.gamepad.set_axis_value(self.vigemclient.Xbox360Axis.LEFT_THUMB_Y, left_y)
        self.gamepad.set_axis_value(self.vigemclient.Xbox360Axis.RIGHT_THUMB_X, right_x)

        # Let the virtual stick pulse briefly; repeated inference calls keep it alive.
        time.sleep(0.02)

    def release_all(self) -> None:
        if self.backend == "vgamepad" and self.pad is not None:
            try:
                self.pad.left_joystick_float(0.0, 0.0)
                self.pad.right_joystick_float(0.0, 0.0)
                self.pad.update()
            except Exception:
                pass
        elif self.backend == "vigemclient" and self.gamepad is not None:
            try:
                self.gamepad.set_axis_value(self.vigemclient.Xbox360Axis.LEFT_THUMB_Y, 0)
                self.gamepad.set_axis_value(self.vigemclient.Xbox360Axis.RIGHT_THUMB_X, 0)
            except Exception:
                pass

    def close(self) -> None:
        self.release_all()
        if self.backend == "vgamepad":
            # Force vgamepad cleanup while module globals are still alive; this avoids
            # noisy ignored exceptions during Python interpreter shutdown.
            try:
                del self.pad
            except Exception:
                pass
            self.pad = None
        elif self.backend == "vigemclient":
            try:
                self.gamepad.unregister()
            finally:
                self.client.disconnect()
