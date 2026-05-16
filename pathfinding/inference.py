"""
inference.py - 行为克隆模型推理
加载训练好的bc_best.pt，实时根据游戏画面+鼠标状态输出寻路动作
"""
import os
import sys
import time
import argparse
import numpy as np

import torch
import dxcam
from pynput import mouse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FRAME_STACK, NUM_ACTIONS, ACTION_SPACE
from models.resnet_encoder import create_encoder
from env.screen_capture import ScreenCapture
from env.action_executor import ActionExecutor


class MouseTracker:
    """pynput鼠标监听器：实时追踪鼠标移动+按键状态"""
    
    def __init__(self, callback):
        self._callback = callback  # (dx, dy, buttons) -> None
        self._listener = None
    
    def _on_move(self, x, y):
        # pynput只报告位置，不报告delta；用相对运动需要系统支持
        # 这里传0，dx/dy由窗口焦点游戏内的鼠标控制
        pass
    
    def _on_click(self, x, y, button, pressed):
        buttons = 0
        if button == mouse.Button.left:
            buttons = 1 if pressed else 0
        elif button == mouse.Button.right:
            buttons = 2 if pressed else 0
        self._callback(0, 0, buttons)
    
    def _on_scroll(self, x, y, dx, dy):
        self._callback(0, 0, 0)
    
    def start(self):
        self._listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self._listener.start()
    
    def stop(self):
        if self._listener:
            self._listener.stop()
            self._listener = None


class BehaviorCloneModel(torch.nn.Module):
    """行为克隆模型：视觉编码器 + 鼠标特征 + 分类头"""
    
    def __init__(self, latent_dim=256, num_actions=NUM_ACTIONS, mouse_dim=4):
        super().__init__()
        self.encoder = create_encoder("resnet18", latent_dim=latent_dim)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + mouse_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, num_actions),
        )
    
    def forward(self, x, mouse_feat):
        feat = self.encoder(x)
        combined = torch.cat([feat, mouse_feat], dim=1)
        return self.classifier(combined)


class PathfindingInference:
    """寻路推理管道"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[推理] 设备: {self.device}", flush=True)
        
        # 动作名称列表
        self.action_names = [a[0] for a in ACTION_SPACE]
        
        # 加载模型
        self.model = BehaviorCloneModel(mouse_dim=4).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"[推理] 模型加载成功: {model_path}", flush=True)
        
        # 帧缓冲区（存储最近FRAME_STACK帧）
        self.frame_buffer = []
        
        # 鼠标状态（由外部更新）
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.mouse_buttons = 0
        
        # 动作执行器
        self.executor = ActionExecutor()
        
        # 画面捕获
        self.capture = ScreenCapture()
        
        # 帧预处理参数
        self.img_size = (224, 224)
        
        # 鼠标追踪器（修复：原来有update_mouse但从未被调用！）
        def mouse_callback(dx, dy, buttons):
            self.update_mouse(dx, dy, buttons)
        self._mouse_tracker = MouseTracker(mouse_callback)
        self._mouse_tracker.start()
        print("[推理] 鼠标监听已启动", flush=True)
        
    def update_mouse(self, dx, dy, buttons):
        """更新鼠标状态（由外部pynput监听器调用）"""
        self.mouse_dx = dx
        self.mouse_dy = dy
        self.mouse_buttons = buttons
    
    def preprocess_frame(self, frame):
        """预处理单帧：resize + normalize + (C,H,W)"""
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
    
    def get_mouse_features(self):
        """获取鼠标特征向量"""
        return np.array([
            np.clip(self.mouse_dx / 100.0, -1.0, 1.0),
            np.clip(self.mouse_dy / 100.0, -1.0, 1.0),
            float((self.mouse_buttons & 1) > 0),  # left
            float((self.mouse_buttons & 2) > 0),  # right
        ], dtype=np.float32)
    
    def predict(self):
        """推理：返回动作ID + 置信度"""
        if len(self.frame_buffer) < FRAME_STACK:
            return None, 0.0
        
        # 堆叠帧 (stack, C, H, W) → (1, stack*C, H, W)
        stacked = np.concatenate(self.frame_buffer, axis=0)
        stacked = np.expand_dims(stacked, axis=0)  # (1, 12, 224, 224)
        
        mouse_feat = self.get_mouse_features()
        mouse_tensor = torch.from_numpy(mouse_feat).unsqueeze(0).to(self.device)
        frame_tensor = torch.from_numpy(stacked).to(self.device)
        
        with torch.no_grad():
            logits = self.model(frame_tensor, mouse_tensor)
            probs = torch.softmax(logits, dim=1)
            action = logits.argmax(1).item()
            confidence = probs[0, action].item()
        
        return action, confidence
    
    def run(self, duration=60, fps=30):
        """运行推理（duration秒，0=无限）"""
        print(f"[推理] 开始运行 (duration={duration}s, fps={fps})", flush=True)
        print(f"[推理] 动作映射: {self.action_names}", flush=True)
        
        start_time = time.time()
        frame_interval = 1.0 / fps
        
        try:
            while True:
                loop_start = time.time()
                
                # 检查时长
                if duration > 0 and (loop_start - start_time) > duration:
                    break
                
                # 捕获画面
                frame = self.capture.grab()
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                # 添加到缓冲区
                self.add_frame(frame)
                
                # 推理
                action, conf = self.predict()
                if action is not None:
                    action_name = self.action_names[action]
                    print(f"[推理] 动作: {action_name} (置信度: {conf:.2%})", flush=True)
                    
                    # 执行动作（idle不执行）
                    if action > 0:  # 0=idle，不执行
                        self.executor.execute(action)
                else:
                    print(f"[推理] 等待帧缓冲区填满 ({len(self.frame_buffer)}/{FRAME_STACK})", flush=True)
                
                # 控制帧率
                elapsed = time.time() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
        
        except KeyboardInterrupt:
            print("\n[推理] 用户中断", flush=True)
        
        finally:
            self._mouse_tracker.stop()
            self.executor.release_all()
            print("[推理] 运行结束", flush=True)


def main():
    parser = argparse.ArgumentParser(description="寻路推理")
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "checkpoints", "bc_best.pt"))
    parser.add_argument("--duration", type=int, default=60, help="运行时长(秒)，0=无限")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"[错误] 模型文件不存在: {args.model}", flush=True)
        return
    
    inferencer = PathfindingInference(args.model)
    inferencer.run(duration=args.duration, fps=args.fps)


if __name__ == "__main__":
    main()
