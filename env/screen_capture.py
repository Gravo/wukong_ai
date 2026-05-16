"""
screen_capture.py - 高速游戏截图模块
支持 dxcam (120fps+) / mss / win32gui 三种后端
"""
import time
import numpy as np
from config import CAPTURE_BACKEND, GAME_REGION, CAPTURE_FPS

class ScreenCapture:
    """游戏画面捕获器"""
    
    def __init__(self, backend=None, region=None, target_fps=None):
        self.backend = backend or CAPTURE_BACKEND
        self.region = region or GAME_REGION
        self.target_fps = target_fps or CAPTURE_FPS
        self._capture = None
        self._last_frame_time = 0
        self._frame_interval = 1.0 / self.target_fps
        
        self._init_backend()
    
    def _init_backend(self):
        """初始化截图后端"""
        if self.backend == "dxcam":
            self._init_dxcam()
        elif self.backend == "mss":
            self._init_mss()
        elif self.backend == "win32":
            self._init_win32()
        else:
            raise ValueError(f"Unknown capture backend: {self.backend}")
    
    def _init_dxcam(self):
        """初始化 dxcam（最快，DXGI方式，120fps+）"""
        try:
            import dxcam
            self._capture = dxcam.create(
                region=(
                    self.region["left"],
                    self.region["top"],
                    self.region["left"] + self.region["width"],
                    self.region["top"] + self.region["height"],
                ),
                output_color="BGR",
            )
            print(f"[ScreenCapture] dxcam 初始化成功，目标 {self.target_fps}fps")
        except ImportError:
            print("[ScreenCapture] dxcam 未安装，回退到 mss")
            self.backend = "mss"
            self._init_mss()
    
    def _init_mss(self):
        """初始化 mss（跨平台，约60fps）"""
        try:
            import mss
            self._mss = mss.mss()
            self._mss_region = {
                "top": self.region["top"],
                "left": self.region["left"],
                "width": self.region["width"],
                "height": self.region["height"],
            }
            print(f"[ScreenCapture] mss 初始化成功")
        except ImportError:
            print("[ScreenCapture] mss 未安装，回退到 win32")
            self.backend = "win32"
            self._init_win32()
    
    def _init_win32(self):
        """初始化 win32gui 截图（最慢，约30fps，但兼容性最好）"""
        import ctypes
        from ctypes import wintypes
        self._user32 = ctypes.windll.user32
        self._gdi32 = ctypes.windll.gdi32
        print(f"[ScreenCapture] win32 初始化成功（最慢后端）")
    
    def grab(self):
        """
        抓取一帧游戏画面
        Returns:
            np.ndarray: BGR格式的图像，shape=(H, W, 3)
        """
        if self.backend == "dxcam":
            return self._grab_dxcam()
        elif self.backend == "mss":
            return self._grab_mss()
        else:
            return self._grab_win32()
    
    def _grab_dxcam(self):
        """dxcam截图"""
        frame = self._capture.grab()
        if frame is None:
            # dxcam偶尔返回None，返回黑帧
            return np.zeros(
                (self.region["height"], self.region["width"], 3),
                dtype=np.uint8
            )
        return frame
    
    def _grab_mss(self):
        """mss截图"""
        import mss
        shot = self._mss.grab(self._mss_region)
        frame = np.array(shot, dtype=np.uint8)
        # mss返回BGRA，去掉alpha通道
        return frame[:, :, :3]
    
    def _grab_win32(self):
        """win32截图"""
        import ctypes
        import win32gui
        import win32ui
        import win32con
        
        hwnd = win32gui.FindWindow(None, "黑神话：悟空")
        if not hwnd:
            hwnd = win32gui.GetDesktopWindow()
        
        # 获取窗口DC
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        
        # 创建位图
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(
            mfc_dc,
            self.region["width"],
            self.region["height"]
        )
        save_dc.SelectObject(bitmap)
        
        # 截图
        save_dc.BitBlt(
            (0, 0),
            (self.region["width"], self.region["height"]),
            mfc_dc,
            (self.region["left"], self.region["top"]),
            win32con.SRCCOPY,
        )
        
        # 转换为numpy
        bmp_info = bitmap.GetInfo()
        bmp_str = bitmap.GetBitmapBits(True)
        frame = np.frombuffer(bmp_str, dtype=np.uint8)
        frame = frame.reshape(
            bmp_info['bmHeight'],
            bmp_info['bmWidth'],
            4
        )[:, :, :3]  # BGR
        
        # 清理
        mfc_dc.DeleteDC()
        save_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        bitmap.DeleteObject()
        
        return frame
    
    def start_video(self, target_fps=None):
        """启动dxcam视频流模式（更高效）"""
        if self.backend == "dxcam" and self._capture:
            fps = target_fps or self.target_fps
            self._capture.start(target_fps=fps, video_mode=True)
            print(f"[ScreenCapture] dxcam 视频流模式启动，{fps}fps")
    
    def stop_video(self):
        """停止dxcam视频流"""
        if self.backend == "dxcam" and self._capture:
            self._capture.stop()
            print("[ScreenCapture] dxcam 视频流已停止")
    
    def get_latest_frame(self):
        """获取视频流模式下的最新帧"""
        if self.backend == "dxcam" and self._capture:
            frame = self._capture.get_latest_frame()
            if frame is None:
                return np.zeros(
                    (self.region["height"], self.region["width"], 3),
                    dtype=np.uint8
                )
            return frame
        return self.grab()
    
    def release(self):
        """释放资源"""
        if self.backend == "dxcam" and self._capture:
            self._capture.stop()
            del self._capture
            self._capture = None
        elif self.backend == "mss" and hasattr(self, '_mss'):
            self._mss.close()
        print("[ScreenCapture] 资源已释放")


# 全局单例
_capture_instance = None

def get_capture(backend=None, region=None):
    """获取全局截图器单例"""
    global _capture_instance
    if _capture_instance is None:
        _capture_instance = ScreenCapture(backend=backend, region=region)
    return _capture_instance
