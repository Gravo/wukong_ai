"""
dagger_collector.py - DAgger 数据采集器 v1.0
AI推理 + 人工干预 + 纠正数据记录

Usage:
    C:\Python\python.exe -u dagger_collector.py ^
      --model checkpoints\goal_bc_epoch_030.pt ^
      --goal-id 1 --output pathfinding_dagger_01.h5 ^
      --fps 10 --emu-alpha 0.3
"""
import argparse, time, os, h5py, json, glob
import numpy as np
import torch
import cv2
import dxcam
import pydirectinput as pdi
import keyboard as kb

from training.goal_conditioned_bc import GoalConditionedBC

# 动作定义（与data_collector_v3.py一致）
ACTION_NAMES = ["idle","attack","heavy","dodge","forward","right","left","dodge_atk","lock","heal"]
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
    """预处理游戏画面"""
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    frame = (frame - mean) / std
    frame = frame.transpose(2, 0, 1)
    return torch.from_numpy(frame).unsqueeze(0)


def execute_action(action_id, mouse_dx, mouse_dy, pixels_per_unit):
    """执行动作"""
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
    for key_name, key_code in [("w", 0x57), ("a", 0x41), ("d", 0x44), 
                                ("j", 0x4A), ("k", 0x4B), ("space", 0x20),
                                ("r", 0x52), ("v", 0x56)]:
        if kb.is_pressed(key_name):
            return ACTION_NAMES.index(key_name) if key_name in ACTION_NAMES else None
    return 0  # 默认idle


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    print(f"[DAgger] Loading model: {args.model}", flush=True)
    checkpoint = torch.load(args.model, map_location='cpu')
    num_goals = checkpoint['goal_embed.weight'].shape[0]
    model = GoalConditionedBC(num_goals=num_goals).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"[DAgger] Model loaded. num_goals={num_goals}", flush=True)

    # 2. 初始化摄像头
    print(f"[DAgger] Initializing camera...", flush=True)
    camera = dxcam.create(output_color="BGR", region=(0, 0, 1920, 1080))
    camera.start(region=(0, 0, 1920, 1080), target_fps=args.fps)
    print(f"[DAgger] Camera started. FPS={args.fps}", flush=True)

    # 3. 初始化数据记录
    frames_buffer = []
    ai_actions = []  # AI预测的动作
    human_actions = []  # 人工纠正的动作（如果有）
    mouse_dx_buffer = []
    mouse_dy_buffer = []
    intervention_flags = []  # 标记是否为干预帧

    mouse_smoother = EMASmoother(alpha=args.ema_alpha)
    goal_id = torch.tensor([args.goal_id], dtype=torch.long).to(device)

    print(f"\n[DAgger] Started! Press 'Q' to toggle intervention mode.", flush=True)
    print(f"[DAgger] Goal ID: {args.goal_id}", flush=True)
    print(f"[DAgger] Output: {args.output}\n", flush=True)

    # 4. 主循环
    intervention_mode = False
    frame_count = 0
    intervention_count = 0

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # 检测 'Q' 键切换干预模式
            if kb.is_pressed('q'):
                intervention_mode = not intervention_mode
                if intervention_mode:
                    print(f"\n[DAgger] ⚠️  Intervention mode ON. Human taking over...", flush=True)
                else:
                    print(f"\n[DAgger] ✅ Intervention mode OFF. AI resuming...", flush=True)
                time.sleep(0.5)  # 防抖

            # 预处理帧
            input_tensor = preprocess_frame(frame).to(device)

            if not intervention_mode:
                # AI模式：推理 + 执行
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
                # 人工干预模式：记录人工操作
                human_action_id = detect_human_action()

                # 简单鼠标追踪（这里可以改进：实际应该记录人类鼠标移动）
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

        # 5. 保存数据
        print(f"\n[DAgger] Saving data to {args.output}...", flush=True)
        save_dagger_data(args.output, frames_buffer, ai_actions, human_actions,
                       mouse_dx_buffer, mouse_dy_buffer, intervention_flags,
                       args.goal_id, is_checkpoint=False)
        print(f"[DAgger] ✅ Data saved. Total frames: {frame_count}, Interventions: {intervention_count}", flush=True)


def save_dagger_data(output_path, frames, ai_actions, human_actions,
                     mouse_dx, mouse_dy, intervention_flags, goal_id, is_checkpoint):
    """保存DAgger数据到h5文件"""
    mode = "a" if is_checkpoint and os.path.exists(output_path) else "w"
    with h5py.File(output_path, mode) as f:
        if "frames" not in f:
            f.create_dataset("frames", data=np.array(frames), compression="gzip")
            f.create_dataset("ai_actions", data=np.array([-1 if a is None else a for a in ai_actions]), compression="gzip")
            f.create_dataset("human_actions", data=np.array([-1 if a is None else a for a in human_actions]), compression="gzip")
            f.create_dataset("mouse_dx", data=np.array(mouse_dx), compression="gzip")
            f.create_dataset("mouse_dy", data=np.array(mouse_dy), compression="gzip")
            f.create_dataset("intervention", data=np.array(intervention_flags), compression="gzip")
            f.attrs["goal_id"] = goal_id
            f.attrs["timestamp"] = int(time.time())
        else:
            # 追加模式（checkpoint）
            f["frames"].resize((f["frames"].shape[0] + len(frames)), axis=0)
            f["frames"][-len(frames):] = np.array(frames)
            f["ai_actions"].resize((f["ai_actions"].shape[0] + len(ai_actions)), axis=0)
            f["ai_actions"][-len(ai_actions):] = np.array([-1 if a is None else a for a in ai_actions])
            f["human_actions"].resize((f["human_actions"].shape[0] + len(human_actions)), axis=0)
            f["human_actions"][-len(human_actions):] = np.array([-1 if a is None else a for a in human_actions])
            f["mouse_dx"].resize((f["mouse_dx"].shape[0] + len(mouse_dx)), axis=0)
            f["mouse_dx"][-len(mouse_dx):] = np.array(mouse_dx)
            f["mouse_dy"].resize((f["mouse_dy"].shape[0] + len(mouse_dy)), axis=0)
            f["mouse_dy"][-len(mouse_dy):] = np.array(mouse_dy)
            f["intervention"].resize((f["intervention"].shape[0] + len(intervention_flags)), axis=0)
            f["intervention"][-len(intervention_flags):] = np.array(intervention_flags)


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
