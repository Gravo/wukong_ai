"""
inference_goal.py - Goal-Conditioned BC 推理 v3.0
- tanh+可学习缩放的鼠标头
- EMA 鼠标平滑
- 模型输出已含 mouse_scale，推理只需乘 pixels_per_unit

Usage:
    C:\Python\python.exe -u training\inference_goal.py ^
      --model checkpoints\goal_bc_epoch_010.pt ^
      --goal-id 1 --duration 60 --fps 10 --pixels-per-unit 50
"""
import argparse, time, numpy as np, torch, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goal_conditioned_bc import GoalConditionedBC

import dxcam
import pydirectinput as pdi

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
    import cv2
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    frame = (frame - mean) / std
    frame = frame.transpose(2, 0, 1)
    return torch.from_numpy(frame).unsqueeze(0)


def execute_action(action_id, mouse_dx, mouse_dy, pixels_per_unit):
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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model, map_location='cpu')
    num_goals = checkpoint['goal_embed.weight'].shape[0]
    print(f"[Inference] Auto-detected num_goals={num_goals}", flush=True)

    model = GoalConditionedBC(num_goals=num_goals).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"[Inference] Model: {args.model}", flush=True)
    print(f"[Inference] Goal ID: {args.goal_id}", flush=True)
    print(f"[Inference] Pixels per unit: {args.pixels_per_unit}", flush=True)
    print(f"[Inference] EMA alpha: {args.ema_alpha}", flush=True)

    camera = dxcam.create(output_color="BGR", region=(0, 0, 1920, 1080))
    camera.start(region=(0, 0, 1920, 1080), target_fps=args.fps)
    print(f"[Inference] Camera started, FPS={args.fps}", flush=True)

    goal_id = torch.tensor([args.goal_id], dtype=torch.long).to(device)
    mouse_smoother = EMASmoother(alpha=args.ema_alpha)

    start_time = time.time()
    frame_count = 0

    print(f"\n[Inference] Starting... Press Ctrl+C to stop\n", flush=True)

    try:
        while (time.time() - start_time) < args.duration:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            input_tensor = preprocess_frame(frame).to(device)

            with torch.no_grad():
                action_logits, mouse_pred = model(input_tensor, goal_id)
                action_id = torch.argmax(action_logits, dim=1).item()
                raw_mouse = mouse_pred.cpu().numpy()[0]

            smoothed_mouse = mouse_smoother.update(raw_mouse)
            execute_action(action_id, smoothed_mouse[0], smoothed_mouse[1], args.pixels_per_unit)

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"[Inference] Frame {frame_count}: action={ACTION_NAMES[action_id]} "
                      f"mouse=({smoothed_mouse[0]:.3f}, {smoothed_mouse[1]:.3f}) "
                      f"raw=({raw_mouse[0]:.3f}, {raw_mouse[1]:.3f})", flush=True)

            time.sleep(1.0 / args.fps)

    except KeyboardInterrupt:
        print("\n[Inference] Stopped by user", flush=True)

    finally:
        camera.stop()
        print(f"\n[Inference] Total frames: {frame_count}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--goal-id", type=int, default=0)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--pixels-per-unit", type=float, default=50,
                        help="Pixels per unit of model mouse output (training normalized by std, so this should be ~avg_mouse_std)")
    args = parser.parse_args()
    main(args)
