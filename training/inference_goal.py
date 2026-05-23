"""
inference_goal.py - Real-time inference with trained Goal-Conditioned BC model
Usage:
    C:\Python\python.exe -u training\inference_goal.py --model checkpoints\goal_bc_epoch_015.pt --goal-id 0 --duration 300 --fps 10
"""
import argparse, time, numpy as np, torch, h5py, os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goal_conditioned_bc import GoalConditionedBC

import dxcam
import pydirectinput as pdi

ACTION_NAMES = ["idle","attack","heavy","dodge","forward","right","left","dodge_atk","lock","heal"]
ACTION_KEYS = {
    0: None,  # idle
    1: "j",  # attack
    2: "k",  # heavy
    3: "space",  # dodge
    4: "w",  # forward
    5: "d",  # right
    6: "a",  # left
    7: "j",  # dodge_atk (combo)
    8: "r",  # lock
    9: "v",  # heal
}

def preprocess_frame(frame):
    """Preprocess frame to (1, 3, 224, 224)"""
    import cv2
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    frame = (frame - mean) / std
    frame = frame.transpose(2, 0, 1)  # (224, 224, 3) -> (3, 224, 224)
    return torch.from_numpy(frame).unsqueeze(0)  # (1, 3, 224, 224)

def execute_action(action_id, mouse_dx, mouse_dy):
    """Execute predicted action"""
    # Mouse movement
    dx = int(mouse_dx * 50)  # scale factor
    dy = int(mouse_dy * 50)
    if abs(dx) > 1 or abs(dy) > 1:
        pdi.moveRel(dx, dy, relative=True)
    
    # Keyboard action
    key = ACTION_KEYS.get(action_id)
    if key and key != "space":
        pdi.keyDown(key)
        time.sleep(0.05)
        pdi.keyUp(key)
    elif key == "space":
        pdi.press(key)

def main(args):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoalConditionedBC(num_goals=1).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"[Inference] Model loaded: {args.model}", flush=True)
    print(f"[Inference] Device: {device}", flush=True)
    print(f"[Inference] Goal ID: {args.goal_id}", flush=True)
    
    # Setup camera
    camera = dxcam.create(output_color="BGR", region=(0, 0, 1920, 1080))
    camera.start(region=(0, 0, 1920, 1080), target_fps=args.fps)
    print(f"[Inference] Camera started, FPS={args.fps}", flush=True)
    
    goal_id = torch.tensor([args.goal_id], dtype=torch.long).to(device)
    
    start_time = time.time()
    frame_count = 0
    
    print(f"\n[Inference] Starting... Press Ctrl+C to stop\n", flush=True)
    
    try:
        while (time.time() - start_time) < args.duration:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Preprocess
            input_tensor = preprocess_frame(frame).to(device)
            
            # Inference
            with torch.no_grad():
                action_logits, mouse_pred = model(input_tensor, goal_id)
                action_id = torch.argmax(action_logits, dim=1).item()
                mouse_dx, mouse_dy = mouse_pred.cpu().numpy()[0]
            
            # Execute
            execute_action(action_id, mouse_dx, mouse_dy)
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"[Inference] Frame {frame_count}: action={ACTION_NAMES[action_id]} "
                      f"mouse=({mouse_dx:.2f}, {mouse_dy:.2f})", flush=True)
            
            time.sleep(1.0 / args.fps)
    
    except KeyboardInterrupt:
        print("\n[Inference] Stopped by user", flush=True)
    
    finally:
        camera.stop()
        print(f"\n[Inference] Total frames: {frame_count}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--goal-id", type=int, default=0, help="Goal ID (default 0)")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=10, help="Inference FPS")
    args = parser.parse_args()
    main(args)