"""
inference_v2.py - 行为克隆模型推理v2
鼠标是输出！模型同时预测WASD动作和鼠标dx/dy，推理时执行两者。

加载训练好的bc_best.pt（v2模型），实时根据游戏画面输出：
  - 离散动作（WASD等）→ 按键执行
  - 连续鼠标移动（dx, dy）→ 鼠标相对移动执行
"""
import os
import sys
import time
import argparse
import numpy as np

import torch
import dxcam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FRAME_STACK, NUM_ACTIONS, ACTION_SPACE
from models.resnet_encoder import create_encoder
from env.screen_capture import ScreenCapture
from env.action_executor import ActionExecutor


class BehaviorCloneModel(torch.nn.Module):
    """行为克隆模型v2：视觉编码器 → 分类头 + 回归头"""
    
    def __init__(self, latent_dim=256, num_actions=NUM_ACTIONS):
        super().__init__()
        self.encoder = create_encoder("resnet18", latent_dim=latent_dim)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, num_actions),
        )
        self.mouse_regressor = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )
    
    def forward(self, x):
        feat = self.encoder(x)
        action_logits = self.classifier(feat)
        mouse_pred = self.mouse_regressor(feat)
        return action_logits, mouse_pred


class PathfindingInference:
    """寻路推理管道v2"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[推理] 设备: {self.device}", flush=True)
        
        self.action_names = [a[0] for a in ACTION_SPACE]
        
        # 加载模型
        self.model = BehaviorCloneModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"[推理] 模型加载成功: {model_path}", flush=True)
        
        # 帧缓冲区
        self.frame_buffer = []
        
        # 动作执行器
        self.executor = ActionExecutor()
        
        # 画面捕获
        self.capture = ScreenCapture()
        
        # 帧预处理参数
        self.img_size = (224, 224)
        
        # 鼠标移动缩放：训练时dx/dy归一化到[-1,1]，需要映射到像素
        self.mouse_scale = 100  # dx=1.0 → 移动100像素
        
    def preprocess_frame(self, frame):
        """预处理单帧"""
        import cv2
        frame = cv2.resize(frame, self.img_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        return frame
    
    def add_frame(self, frame):
        """添加一帧到缓冲区"""
        processed = self.preprocess_frame(frame)
        self.frame_buffer.append(processed)
        if len(self.frame_buffer) > FRAME_STACK:
            self.frame_buffer.pop(0)
    
    def predict(self):
        """推理：返回 (动作ID, 置信度, mouse_dx, mouse_dy)"""
        if len(self.frame_buffer) < FRAME_STACK:
            return None, 0.0, 0.0, 0.0
        
        stacked = np.concatenate(self.frame_buffer, axis=0)
        stacked = np.expand_dims(stacked, axis=0)
        frame_tensor = torch.from_numpy(stacked).to(self.device)
        
        with torch.no_grad():
            action_logits, mouse_pred = self.model(frame_tensor)
            probs = torch.softmax(action_logits, dim=1)
            action = action_logits.argmax(1).item()
            confidence = probs[0, action].item()
            dx = mouse_pred[0, 0].item()
            dy = mouse_pred[0, 1].item()
        
        return action, confidence, dx, dy
    
    def execute_mouse(self, dx, dy):
        """执行鼠标相对移动（dx/dy是归一化值[-1,1]）"""
        import pydirectinput
        pixel_dx = int(dx * self.mouse_scale)
        pixel_dy = int(dy * self.mouse_scale)
        if abs(pixel_dx) > 0 or abs(pixel_dy) > 0:
            pydirectinput.moveRel(pixel_dx, pixel_dy, relative=True)
    
    def _detect_game_window(self):
        """检测游戏窗口位置并返回区域dict"""
        import ctypes
        from ctypes import wintypes, c_long
        user32 = ctypes.windll.user32
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        class RECT(ctypes.Structure):
            _fields_ = [("left", c_long), ("top", c_long), ("right", c_long), ("bottom", c_long)]
        result = None
        def enum_cb(hwnd, lparam):
            nonlocal result
            if user32.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, buf, 256)
                title = buf.value.lower()
                if any(k in title for k in ['黑神话', 'wukong', 'b1']):
                    r = RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(r))
                    result = {"top": r.top, "left": r.left, "width": r.right - r.left, "height": r.bottom - r.top}
                    return False  # 找到就停
            return True
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        return result
    
    def _focus_game_window(self):
        """查找并激活游戏窗口"""
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        game_hwnd = None
        def enum_cb(hwnd, lparam):
            nonlocal game_hwnd
            if user32.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, buf, 256)
                title = buf.value.lower()
                if any(k in title for k in ['黑神话', 'wukong', 'b1']):
                    game_hwnd = hwnd
                    return False
            return True
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        if game_hwnd:
            user32.SetForegroundWindow(game_hwnd)
            print(f"[推理] 已激活游戏窗口: hwnd={game_hwnd}", flush=True)
            import time; time.sleep(0.5)
        else:
            print("[推理] 警告: 未找到游戏窗口！", flush=True)
    
    def run(self, duration=60, fps=30):
        """运行推理"""
        # 自动检测游戏窗口位置
        game_region = self._detect_game_window()
        if game_region:
            print(f"[推理] 检测到游戏窗口: left={game_region['left']}, top={game_region['top']}, "
                  f"{game_region['width']}x{game_region['height']}", flush=True)
            # 重新创建截图器，使用正确的窗口区域
            self.capture = ScreenCapture(region=game_region)
        else:
            print("[推理] 警告: 未检测到游戏窗口，使用默认区域", flush=True)
        
        self._focus_game_window()
        print(f"[推理] 开始运行 (duration={duration}s, fps={fps})", flush=True)
        print(f"[推理] 动作映射: {self.action_names}", flush=True)
        
        start_time = time.time()
        frame_interval = 1.0 / fps
        
        try:
            while True:
                loop_start = time.time()
                
                if duration > 0 and (loop_start - start_time) > duration:
                    break
                
                frame = self.capture.grab()
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                self.add_frame(frame)
                
                action, conf, dx, dy = self.predict()
                if action is not None:
                    action_name = self.action_names[action]
                    print(
                        f"[推理] 动作: {action_name} ({conf:.2%}) | "
                        f"鼠标: dx={dx:.2f}, dy={dy:.2f}",
                        flush=True,
                    )
                    
                    # 执行按键动作
                    if action > 0:
                        self.executor.execute(action)
                    
                    # 执行鼠标移动
                    self.execute_mouse(dx, dy)
                else:
                    print(f"[推理] 等待帧缓冲区填满 ({len(self.frame_buffer)}/{FRAME_STACK})", flush=True)
                
                elapsed = time.time() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
        
        except KeyboardInterrupt:
            print("\n[推理] 用户中断", flush=True)
        
        finally:
            self.executor.release_all()
            print("[推理] 运行结束", flush=True)


def main():
    parser = argparse.ArgumentParser(description="寻路推理v2（鼠标是输出）")
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "checkpoints", "bc_best.pt"))
    parser.add_argument("--duration", type=int, default=60, help="运行时长(秒)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mouse-scale", type=int, default=100, help="鼠标移动缩放(像素)")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"[错误] 模型文件不存在: {args.model}", flush=True)
        return
    
    inferencer = PathfindingInference(args.model)
    inferencer.mouse_scale = args.mouse_scale
    inferencer.run(duration=args.duration, fps=args.fps)


if __name__ == "__main__":
    main()
