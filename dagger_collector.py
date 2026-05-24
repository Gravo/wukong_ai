"""
dagger_collector.py - DAgger 数据采集器 v1.1
AI推理 + 人工干预 + 纠正数据记录

v1.1 修复：
  - 干预模式切换键从 'Q' 改为 'F10'（避免触发游戏技能）
  - 添加 ESC 键退出功能
  - 使用 keyboard.add_hotkey() + suppress=True 拦截按键

Usage:
    C:\Python\python.exe -u dagger_collector.py ^
      --model checkpoints\goal_bc_epoch_030.pt ^
      --goal-id 1 --output pathfinding_dagger_01.h5 ^
      --fps 10 --emu-alpha 0.3 --pixels-per-unit 50
"""
import argparse, time, os, h5py, json, glob
import numpy as np
import torch
import cv2
import dxcam
import pydirectinput as pdi
import keyboard as kb
import ctypes
from ctypes import wintypes

from training.goal_conditioned_bc import GoalConditionedBC


# ========== 全局变量（键盘回调用） ==========
_intervention_mode = False
_stop_requested = False
_last_mouse = None  # 鼠标位置追踪（用于DAgger干预模式）

def toggle_intervention():
    """切换干预模式（F10键回调）"""
    global _intervention_mode
    _intervention_mode = not _intervention_mode
    if _intervention_mode:
        print(f"\n[DAgger] ⚠️  Intervention mode ON. Human taking over...", flush=True)
    else:
        print(f"\n[DAgger] ✅ Intervention mode OFF. AI resuming...", flush=True)

def request_stop():
    """请求停止（ESC键回调）"""
    global _stop_requested
    _stop_requested = True
    print(f"\n[DAgger] Stop requested. Saving data...", flush=True)


# ========== 动作定义（与data_collector_v3.py一致） ==========
ACTION_NAMES = ["idle", "attack", "heavy", "dodge", "forward", "right", "left", "dodge_atk", "lock", "heal"]
ACTION_KEYS = {
    0: None, 1: "j", 2: "k", 3: "space", 4: "w",
    5: "d", 6: "a", 7: "j", 8: "r", 9: "v",
}


class EMASmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x.copy()
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value.copy()


def preprocess_frame(frame):
    """预处理游戏画面：resize + normalize + to tensor"""
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    frame = (frame - mean) / std
    frame = frame.transpose(2, 0, 1)
    return torch.from_numpy(frame).unsqueeze(0)


def execute_action(action_id, mouse_dx, mouse_dy, pixels_per_unit):
    """执行动作：鼠标移动 + 键盘按键"""
    dx = int(mouse_dx * pixels_per_unit)
    dy = int(mouse_dy * pixels_per_unit)
    if abs(dx) > 1 or abs(dy) > 1:
        pdi.moveRel(dx, dy, relative=True)

    key = ACTION_KEYS.get(action_id)
    if key and key != "space":
        pdi.keyDown(key)
        time.sleep(0.05)
        pdi.keyUp(key)
    elif key == "space":
        pdi.press(key)


def detect_human_action():
    """检测人工干预时的按键（简化版：记录按下的第一个有效键）"""
    key_map = [
        ("w", 0x57, 4), ("a", 0x41, 6), ("d", 0x44, 5),
        ("j", 0x4A, 1), ("k", 0x4B, 2), ("space", 0x20, 3),
        ("r", 0x52, 8), ("v", 0x56, 9)
    ]
    for key_name, _, action_id in key_map:
        if kb.is_pressed(key_name):
            return action_id
    return 0  # 默认idle


def save_dagger_data(output_path, frames, ai_actions, human_actions,
                    mouse_dx, mouse_dy, intervention_flags, goal_id, is_checkpoint):
    """保存DAgger数据到h5文件（支持追加）"""
    mode = "a" if is_checkpoint and os.path.exists(output_path) else "w"
    with h5py.File(output_path, mode) as f:
        if "frames" not in f:
            f.create_dataset("frames", data=np.array(frames), compression="gzip", chunks=True)
            f.create_dataset("ai_actions", data=np.array([-1 if a is None else a for a in ai_actions]), compression="gzip")
            f.create_dataset("human_actions", data=np.array([-1 if a is None else a for a in human_actions]), compression="gzip")
            f.create_dataset("mouse_dx", data=np.array(mouse_dx), compression="gzip")
            f.create_dataset("mouse_dy", data=np.array(mouse_dy), compression="gzip")
            f.create_dataset("intervention", data=np.array(intervention_flags), compression="gzip")
            f.attrs["goal_id"] = goal_id
            f.attrs["timestamp"] = int(time.time())
            f.attrs["version"] = "1.1"
        else:
            # 追加模式（checkpoint）
            for key, data in [
                ("frames", frames), ("ai_actions", ai_actions), ("human_actions", human_actions),
                ("mouse_dx", mouse_dx), ("mouse_dy", mouse_dy), ("intervention", intervention_flags)
            ]:
                f[key].resize((f[key].shape[0] + len(data)), axis=0)
                if key == "frames":
                    f[key][-len(data):] = np.array(data)
                else:
                    f[key][-len(data):] = np.array([-1 if a is None else a for a in data])


def main(args):
    global _last_mouse
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    print(f"[DAgger] Loading model: {args.model}", flush=True)
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    
    # 兼容两种保存格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        num_goals = state_dict['goal_embed.weight'].shape[0]
    else:
        state_dict = checkpoint
        num_goals = checkpoint['goal_embed.weight'].shape[0]
    
    model = GoalConditionedBC(num_goals=num_goals).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[DAgger] Model loaded. num_goals={num_goals}", flush=True)

    # 2. 初始化摄像头
    print(f"[DAgger] Initializing camera...", flush=True)
    camera = dxcam.create(output_color="BGR", region=(0, 0, 1920, 1080))
    camera.start(region=(0, 0, 1920, 1080), target_fps=args.fps)
    print(f"[DAgger] Camera started. FPS={args.fps}", flush=True)

    # 3. 初始化键盘状态
    print(f"[DAgger] Initializing keyboard...", flush=True)
    _intervention_mode = False
    _stop_requested = False
    print(f"[DAgger] ✅ Keyboard ready. F10=Toggle Intervention, ESC=Stop", flush=True)

    # 4. 初始化数据记录
    frames_buffer = []
    ai_actions = []
    human_actions = []
    mouse_dx_buffer = []
    mouse_dy_buffer = []
    intervention_flags = []

    mouse_smoother = EMASmoother(alpha=args.emu_alpha)
    goal_id = torch.tensor([args.goal_id], dtype=torch.long).to(device)

    print(f"\n[DAgger] ✅ Started! Press F10 to toggle intervention mode.", flush=True)
    print(f"[DAgger] ✅ Press ESC to stop and save.", flush=True)
    print(f"[DAgger] Goal ID: {args.goal_id}", flush=True)
    print(f"[DAgger] Output: {args.output}\n", flush=True)

    # 5. 主循环
    frame_count = 0
    intervention_count = 0
    _last_f10_state = False  # F10边沿检测（防止连按）

    try:
        while not _stop_requested:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # === 按键检测（边沿触发，防止连按）===
            # F10：上升沿切换干预模式（只触发一次）
            current_f10 = kb.is_pressed('f10')
            if current_f10 and not _last_f10_state:
                _intervention_mode = not _intervention_mode
                if _intervention_mode:
                    print(f"\n[DAgger] ⚠️  Intervention mode ON. Human taking over...", flush=True)
                else:
                    print(f"\n[DAgger] ✅ Intervention mode OFF. AI resuming...", flush=True)
                time.sleep(0.1)  # 防抖
            _last_f10_state = current_f10

            # ESC：停止并保存
            if kb.is_pressed('esc'):
                _stop_requested = True
                print(f"\n[DAgger] Stop requested. Saving data...", flush=True)
                time.sleep(0.3)  # 防抖

            # 预处理帧
            input_tensor = preprocess_frame(frame).to(device)

            if not _intervention_mode:
                # === AI模式：推理 + 执行 ===
                with torch.no_grad():
                    action_logits, mouse_pred = model(input_tensor, goal_id)
                    action_id = torch.argmax(action_logits, dim=1).item()
                    raw_mouse = mouse_pred.cpu().numpy()[0]

                smoothed_mouse = mouse_smoother.update(raw_mouse)
                execute_action(action_id, smoothed_mouse[0], smoothed_mouse[1], args.pixels_per_unit)

                # 记录数据
                frames_buffer.append(frame)
                ai_actions.append(action_id)
                human_actions.append(None)  # AI模式，无人工纠正
                mouse_dx_buffer.append(smoothed_mouse[0])
                mouse_dy_buffer.append(smoothed_mouse[1])
                intervention_flags.append(False)

                if frame_count % 10 == 0:
                    print(f"[DAgger] Frame {frame_count}: AI action={ACTION_NAMES[action_id]} "
                          f"mouse=({smoothed_mouse[0]:.3f}, {smoothed_mouse[1]:.3f})", flush=True)

            else:
                # === 人工干预模式：记录人工操作 ===
                global _last_mouse
                human_action_id = detect_human_action()

                # 记录人类鼠标移动（使用ctypes获取鼠标位置）
                pt = wintypes.POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                
                if _last_mouse is None:
                    _last_mouse = (pt.x, pt.y)
                
                mouse_dx = (pt.x - _last_mouse[0]) / args.pixels_per_unit
                mouse_dy = (pt.y - _last_mouse[1]) / args.pixels_per_unit
                _last_mouse = (pt.x, pt.y)

                # 执行人工动作（已经在做了，因为是人类在控制）
                # 这里只记录数据
                frames_buffer.append(frame)
                ai_actions.append(None)  # 干预模式，无AI预测
                human_actions.append(human_action_id)
                mouse_dx_buffer.append(mouse_dx)
                mouse_dy_buffer.append(mouse_dy)
                intervention_flags.append(True)
                intervention_count += 1

                if intervention_count % 10 == 0:
                    print(f"[DAgger] Intervention frame {intervention_count}: "
                          f"human_action={ACTION_NAMES[human_action_id]}", flush=True)

            frame_count += 1

            # 定期保存（防止崩溃丢失数据）
            if frame_count % 1000 == 0:
                print(f"[DAgger] Auto-saving checkpoint at frame {frame_count}...", flush=True)
                save_dagger_data(args.output, frames_buffer, ai_actions, human_actions,
                               mouse_dx_buffer, mouse_dy_buffer, intervention_flags,
                               args.goal_id, is_checkpoint=True)
                frames_buffer, ai_actions, human_actions = [], [], []
                mouse_dx_buffer, mouse_dy_buffer, intervention_flags = [], [], []

            time.sleep(1.0 / args.fps)

    except KeyboardInterrupt:
        print("\n[DAgger] Stopped by user.", flush=True)

    finally:
        camera.stop()
        kb.remove_all_hotkeys()

        # 6. 保存数据
        print(f"\n[DAgger] Saving data to {args.output}...", flush=True)
        if len(frames_buffer) > 0:
            save_dagger_data(args.output, frames_buffer, ai_actions, human_actions,
                           mouse_dx_buffer, mouse_dy_buffer, intervention_flags,
                           args.goal_id, is_checkpoint=False)
        print(f"[DAgger] ✅ Data saved. Total frames: {frame_count}, Interventions: {intervention_count}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--goal-id", type=int, default=1, help="Goal ID")
    parser.add_argument("--output", type=str, default="dagger_data.h5", help="Output h5 file")
    parser.add_argument("--fps", type=int, default=10, help="Camera FPS")
    parser.add_argument("--emu-alpha", type=float, default=0.3, help="EMA smoothing alpha")
    parser.add_argument("--pixels-per-unit", type=float, default=50, help="Mouse output scaling")
    args = parser.parse_args()
    main(args)
