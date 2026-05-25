"""inference_goal_v52.py - v5.2 Non-uniform Frame Stacking Inference

Matches training logic:
- Non-uniform stacking: [0, +1, +3, +7] frames
- Mouse speed quantization for action selection
- Zero-padding for boundary frames
- Goal-conditioned prediction
"""

import os, sys, argparse, time
import numpy as np
import torch, torch.nn as nn
import torchvision.models as models
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_FRAMES = 4
STACK_OFFSETS = [0, 1, 3, 7]
SPEED_THRESHOLDS = [30.0, 150.0, 500.0]
ACTION_NAMES = ["idle", "forward", "turn_slow", "turn_medium", "turn_fast"]

# Mouse pixel deltas per action class per inference tick (1 second)
MOUSE_DELTAS = {
    0: 0,     # idle
    1: 0,     # forward
    2: 40,    # turn_slow  (slight nudge)
    3: 120,   # turn_medium (normal turn)
    4: 300,   # turn_fast  (sharp turn)
}

# Direction tracking: +1 = right, -1 = left
DIRECTIONS = {
    2: 0,  # slow: needs history to determine
    3: 0,  # medium: needs history
    4: 0,  # fast: needs history
}


class V52Model(nn.Module):
    def __init__(self, num_goals, num_frames=NUM_FRAMES, pretrained=False):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3 * num_frames, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.goal_embed = nn.Embedding(num_goals, 64)
        self.fc = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )
    
    def forward(self, x, goal_ids):
        feat = self.backbone(x).flatten(1)
        goal_feat = self.goal_embed(goal_ids)
        return self.fc(torch.cat([feat, goal_feat], dim=1))


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


def execute_action(pred_class, last_mouse_dx):
    """Execute predicted action. Returns mouse dx for direction tracking."""
    import pyautogui
    
    # Release forward
    pyautogui.keyUp('w')
    
    if pred_class == 0:
        # idle: do nothing
        return last_mouse_dx  # preserve direction context
    elif pred_class == 1:
        # forward
        pyautogui.keyDown('w')
        return last_mouse_dx  # preserve direction context
    elif pred_class in (2, 3, 4):
        # Turning: move mouse in the same direction as last turn
        delta = MOUSE_DELTAS[pred_class]
        # Use sign of last_mouse_dx; default right (+1) if never moved
        direction = 1 if last_mouse_dx >= 0 else -1
        pyautogui.move(delta * direction, 0)
        pyautogui.keyDown('w')
        return delta * direction
    return last_mouse_dx


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Detect num_goals from checkpoint
    model = V52Model(num_goals=args.num_goals).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    print(f"[v5.2] Model loaded: {args.model}")
    print(f"[v5.2] Stack offsets: {STACK_OFFSETS}")
    print(f"[v5.2] Goal ID: {args.goal_id}, Duration: {args.duration}s")
    print(f"[v5.2] Press Ctrl+C to stop\n")
    
    frame_buffer = []  # stores preprocessed frames
    last_mouse_dx = 0
    inference_dt = 1.0 / args.fps  # match training fps
    
    for t_tick in range(int(args.duration * args.fps)):
        t0 = time.time()
        
        # Capture screen
        raw = get_screen()
        processed = preprocess(raw)
        frame_buffer.append(processed)
        
        # Keep buffer large enough for max offset
        max_offset = max(STACK_OFFSETS)
        while len(frame_buffer) > max_offset + 1:
            frame_buffer.pop(0)
        
        # Build stacked input with non-uniform offsets
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
                logits = model(x, gid)
                pred_class = torch.argmax(logits, dim=1).item()
            
            last_mouse_dx = execute_action(pred_class, last_mouse_dx)
            
            if t_tick % args.fps == 0:
                print(f"  [{t_tick // args.fps}s] {ACTION_NAMES[pred_class]}")
        
        # Maintain target FPS
        elapsed = time.time() - t0
        sleep_time = max(0, inference_dt - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # Cleanup
    import pyautogui
    pyautogui.keyUp('w')
    try:
        if hasattr(get_screen, '_camera'):
            get_screen._camera.release()
    except Exception:
        pass
    print(f"\n[Done] Ran for {args.duration}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="v5.2 Inference")
    p.add_argument("--model", required=True, help="Path to v5.2 checkpoint")
    p.add_argument("--goal-id", type=int, default=0)
    p.add_argument("--num-goals", type=int, default=2)
    p.add_argument("--duration", type=int, default=60)
    p.add_argument("--fps", type=int, default=15, help="Inference FPS (default: 15, match training)")
    a = p.parse_args()
    main(a)
