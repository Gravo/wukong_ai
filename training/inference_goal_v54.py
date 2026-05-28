"""inference_goal_v54.py - v5.4 双头模型推理（高频微步方案）

基于 pydirectinput 高频小步方案：
  - 不用一次性大位移，改用 500Hz 高频微步（每次 1px）
  - 用 duration 控制转向角度
  - 必须管理员权限运行

架构：
  Input: 4帧堆叠 (12通道) + goal_id (0/1)
    ↓
  ResNet18 + Goal Embedding
    ↓
  ┌─────────────────┬─────────────────┐
  │ Action Head     │ Mouse Head       │
  │ (3-class)       │ (7-bucket)      │
  │ 0=forward       │ 速度档位        │
  │ 1=turn_left     │                 │
  │ 2=turn_right    │                 │
  └─────────────────┴─────────────────┘

执行：
  - forward: pyautogui.keyDown('w')
  - turn: 高频微步循环 (约500Hz) + 保持W键
"""

import os, sys, argparse, time, ctypes, threading
import numpy as np
import torch, torch.nn as nn
import torchvision.models as models
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============ 配置 ============
NUM_FRAMES = 4
STACK_OFFSETS = [0, 1, 3, 7]
INFERENCE_FPS = 15
TICK_DURATION = 1.0 / INFERENCE_FPS  # ~0.0667s

# Action classes (3类)
ACTION_CLASSES = ["forward", "turn_left", "turn_right"]

# Mouse bucket -> 速度 (pixels per second)
# Bucket 0/1/2 = 左转(负), 3=直行, 4/5/6 = 右转(正)
BUCKET_SPEED_PPS = {
    0: 800,   # 快速左转
    1: 400,   # 中速左转
    2: 150,   # 慢速左转
    3: 0,     # 直行
    4: 150,   # 慢速右转
    5: 400,   # 中速右转
    6: 800,   # 快速右转
}

# 高频移动参数（500Hz 方案）
_MOVE_STEP = 1          # 每次移动 1 像素
_MOVE_INTERVAL = 0.002  # 2ms = 500Hz
_STOP_FLAG = False


def _high_freq_move(direction, duration):
    """
    高频微步移动线程（500Hz）。
    direction: +1=右转, -1=左转, 0=不动
    duration: 秒
    """
    if direction == 0 or _STOP_FLAG:
        return
    
    import pydirectinput
    pydirectinput.PAUSE = 0
    pydirectinput.FAILSAFE = True
    
    steps = int(duration / _MOVE_INTERVAL)
    dx = direction * _MOVE_STEP
    
    for _ in range(steps):
        if _STOP_FLAG:
            break
        try:
            pydirectinput.moveRel(dx, 0, relative=True)
        except Exception:
            break
        time.sleep(_MOVE_INTERVAL)


def execute_turn(direction, bucket, tick_duration=TICK_DURATION):
    """
    在一个 tick 内执行转向动作（高频微步方案）。
    
    direction: +1=右转, -1=左转
    bucket: 0~6（速度档位）
    tick_duration: 本次 tick 的持续时间
    """
    speed = BUCKET_SPEED_PPS.get(bucket, 0)
    if speed == 0 or direction == 0:
        return  # 直行不动
    
    # 计算本次 tick 内应该移动的像素
    total_pixels = int(speed * tick_duration)
    if total_pixels == 0:
        return
    
    # 用高频微步线程执行（异步，不阻塞主循环）
    t = threading.Thread(target=_high_freq_move, args=(direction, tick_duration))
    t.daemon = True
    t.start()


class DualHeadModel(nn.Module):
    """双头模型：Action Head (3类) + Mouse Head (7-bucket)"""
    
    def __init__(self, num_actions=3, num_mouse_buckets=7, latent_dim=512, hidden_dim=256):
        super().__init__()
        
        self.backbone = models.resnet18(weights=None)
        old_conv1 = self.backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=12,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False
        )
        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = old_conv1.weight
            for i in range(1, 4):
                new_conv1.weight[:, i*3:(i+1)*3, :, :] = old_conv1.weight
        
        self.backbone.conv1 = new_conv1
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.goal_embed = nn.Embedding(2, 32)
        self.feature_dim = 512 + 32
        
        self.action_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_actions),
        )
        self.mouse_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_mouse_buckets),
        )
    
    def forward(self, frames, goal_ids):
        goal_ids = goal_ids.long()
        B = frames.shape[0]
        x = self.backbone(frames).view(B, -1)
        goal_emb = self.goal_embed(goal_ids)
        features = torch.cat([x, goal_emb], dim=-1)
        action_logits = self.action_head(features)
        mouse_logits = self.mouse_head(features)
        return action_logits, mouse_logits


def preprocess(frame):
    """Preprocess single frame: resize + ImageNet normalize."""
    if frame.shape[0] != 224 or frame.shape[1] != 224:
        frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (frame - mean) / std


def get_screen():
    """Capture game screen using dxcam (preferred) or pyautogui fallback."""
    try:
        import dxcam
        if not hasattr(get_screen, '_camera'):
            get_screen._camera = dxcam.create()
        frame = get_screen._camera.grab()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    
    import pyautogui
    frame = np.array(pyautogui.screenshot())
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 4 else frame


def focus_game_window():
    """聚焦黑神话游戏窗口"""
    try:
        from ctypes import wintypes
        
        def enum_callback(hwnd, results):
            title = ctypes.create_unicode_buffer(256)
            ctypes.windll.user32.GetWindowTextW(hwnd, title, 256)
            if 'Black Myth' in (title.value or '') or '黑神话' in (title.value or '') or 'Wukong' in (title.value or ''):
                results.append(hwnd)
            return True
        
        results = []
        EnumWindows = ctypes.windll.user32.EnumWindows
        EnumWindows(ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(enum_callback), ctypes.byref(ctypes.c_long(0)))
        
        if results:
            hwnd = results[0]
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            ctypes.windll.user32.ShowWindow(hwnd, 9)
            return True
    except Exception:
        pass
    return False


def main(args):
    global _STOP_FLAG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载双头模型
    model = DualHeadModel(num_actions=3, num_mouse_buckets=7).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    print(f"[v5.4] 双头模型已加载: {args.model}")
    print(f"[v5.4] 高频微步方案: 500Hz, {_MOVE_STEP}px/次")
    print(f"[v5.4] Stack offsets: {STACK_OFFSETS}")
    print(f"[v5.4] Goal ID: {args.goal_id}, Duration: {args.duration}s")
    print(f"[v5.4] ⚠️  需要管理员权限运行！")
    print(f"[v5.4] 按 Ctrl+C 停止\n")
    
    # 聚焦游戏窗口
    if focus_game_window():
        print("[v5.4] 游戏窗口已聚焦!")
    else:
        print("[v5.4] ⚠️ 未找到游戏窗口，确保游戏正在运行!")
    
    # 倒计时
    print("[v5.4] 5秒后开始... 现在聚焦游戏窗口!")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("[v5.4] 开始!\n")
    
    # 初始化 pydirectinput
    import pydirectinput
    pydirectinput.PAUSE = 0
    pydirectinput.FAILSAFE = True
    
    import pyautogui
    pyautogui.PAUSE = 0
    
    # 帧缓冲
    frame_buffer = []
    max_offset = max(STACK_OFFSETS)
    
    # 统计
    action_counts = {0: 0, 1: 0, 2: 0}
    bucket_counts = {i: 0 for i in range(7)}
    total = 0
    turn_decisions = 0
    
    start_time = time.time()
    
    try:
        for tick in range(int(args.duration * INFERENCE_FPS)):
            t0 = time.time()
            
            # 截图 + 预处理
            raw = get_screen()
            processed = preprocess(raw)
            frame_buffer.append(processed)
            
            while len(frame_buffer) > max_offset + 1:
                frame_buffer.pop(0)
            
            # 构建堆叠输入
            if len(frame_buffer) >= max_offset + 1:
                base_idx = len(frame_buffer) - max_offset - 1
                
                stacked = []
                for offset in STACK_OFFSETS:
                    fi = base_idx + offset
                    if 0 <= fi < len(frame_buffer):
                        stacked.append(frame_buffer[fi].transpose(2, 0, 1))
                    else:
                        stacked.append(np.zeros((3, 224, 224), dtype=np.float32))
                
                x = torch.from_numpy(np.concatenate(stacked, axis=0)).unsqueeze(0).float().to(device)
                gid = torch.tensor([args.goal_id], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    action_logits, mouse_logits = model(x, gid)
                    
                    # Action 预测 (0=forward, 1=left, 2=right)
                    pred_action = torch.argmax(action_logits, dim=1).item()
                    # Mouse bucket 预测 (0~6)
                    pred_bucket = torch.argmax(mouse_logits, dim=1).item()
                
                # 统计
                action_counts[pred_action] += 1
                bucket_counts[pred_bucket] += 1
                total += 1
                
                # 执行动作
                if pred_action == 0:
                    # forward: 保持 W 按住
                    pyautogui.keyDown('w')
                elif pred_action == 1:
                    # turn_left: 高频微步左转 + W
                    execute_turn(direction=-1, bucket=pred_bucket, tick_duration=TICK_DURATION)
                    pyautogui.keyDown('w')
                elif pred_action == 2:
                    # turn_right: 高频微步右转 + W
                    execute_turn(direction=+1, bucket=pred_bucket, tick_duration=TICK_DURATION)
                    pyautogui.keyDown('w')
                
                if pred_action in (1, 2):
                    turn_decisions += 1
                
                if tick % INFERENCE_FPS == 0:
                    elapsed = tick // INFERENCE_FPS
                    speed = BUCKET_SPEED_PPS.get(pred_bucket, 0)
                    direction_str = ["←左转", "→直行", "→右转"][pred_action] if pred_action == 0 else (f"←左转({speed}pps)" if pred_action == 1 else f"→右转({speed}pps)")
                    turn_rate = 100.0 * turn_decisions / total if total > 0 else 0
                    print(f"  [{elapsed}s] {direction_str:20s} | bucket={pred_bucket} | 转向率={turn_rate:.1f}%")
            
            # 维持目标 FPS
            elapsed = time.time() - t0
            sleep_time = max(0, TICK_DURATION - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n[v5.4] 停止中...")
    
    finally:
        # 清理
        _STOP_FLAG = True
        time.sleep(0.1)
        pyautogui.keyUp('w')
        try:
            if hasattr(get_screen, '_camera'):
                get_screen._camera.release()
        except Exception:
            pass
    
    # 打印统计
    print(f"\n[完成] 运行了 {time.time() - start_time:.1f}s")
    if total > 0:
        print(f"\n=== Action 分布 ===")
        for i in range(3):
            cnt = action_counts[i]
            pct = 100.0 * cnt / total
            print(f"  {ACTION_CLASSES[i]:12s}: {cnt:5d} ({pct:5.1f}%)")
        turn_total = action_counts[1] + action_counts[2]
        print(f"  {'Turn total':12s}: {turn_total:5d} ({100.0*turn_total/total:5.1f}%)")
        
        print(f"\n=== Mouse Bucket 分布 ===")
        for i in range(7):
            cnt = bucket_counts[i]
            pct = 100.0 * cnt / total
            speed = BUCKET_SPEED_PPS[i]
            sign = "+" if speed >= 0 else ""
            print(f"  {i}: bucket {i} ({sign}{speed} pps): {cnt:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="v5.4 双头模型推理（高频微步版）")
    p.add_argument("--model", required=True, help="Path to v5.4 checkpoint (.pt)")
    p.add_argument("--goal-id", type=int, default=0, help="Goal ID (0 or 1)")
    p.add_argument("--num-goals", type=int, default=2)
    p.add_argument("--duration", type=int, default=60, help="运行时间(秒)")
    a = p.parse_args()
    main(a)
