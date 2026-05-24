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

from training.goal_conditioned_bc import GoalConditionedBC


# ========== 全局变量（键盘回调用） ==========
_intervention_mode = False
_stop_requested = False

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    print(f"[DAgger] Loading model: {args.model}", flush=True)
    checkpoint = torch.load(args.model, map_location='cpu')
    num_goals = checkpoint['goal_embed.weight'].shape[0]
    model = GoalConditionedBC(num_goals=num_goals).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[DAgger] Model loaded. num_goals={num_goals}", flush=True)

    # 2. 初始化摄像头
    print(f"[DAgger] Initializing camera...", flush=True)
    camera = dxcam.create(output_color="BGR", region=(0, 0, 1920, 1080))
    camera.start(region=(0, 0, 1920, 1080), target_fps=args.fps)
    print(f"[DAgger] Camera started. FPS={args.fps}", flush=True)

    # 3. 注册键盘热键
    print(f"[DAgger] Registering hotkeys...", flush=True)
    kb.add_hotkey('f10', toggle_intervention, suppress=True)  # F10切换干预模式（拦截，不传到游戏）
    kb.add_hotkey('esc', request_stop, suppress=False)  # ESC停止（不需要拦截）
    print(f"[DAgger] ✅ Hotkeys registered: F10=Toggle Intervention, ESC=Stop", flush=True)

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

    try:
        while not _stop_requested:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

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
                human_action_id = detect_human_action()

                # 简单鼠标追踪（可以改进：实际应该记录人类鼠标移动）
                mouse_dx, mouse_dy = 0, 0  # 简化：暂时不记录鼠标

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
                    print(f"[DAgger]