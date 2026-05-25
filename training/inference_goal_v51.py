"""inference_goal_v51.py - v5.1帧堆叠推理
推理时维护4帧历史窗口"""

import os, sys, argparse, numpy as np, torch, torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import pyautogui, time, cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_FRAMES = 4
ACTION_NAMES = ["idle","forward","turn_left","turn_right","dodge"]
MOUSE_DELTA = 86

class V5Model(nn.Module):
    def __init__(self, num_goals, num_frames=NUM_FRAMES, pretrained=False):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3*num_frames, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.goal_embed = nn.Embedding(num_goals, 64)
        self.fc = nn.Sequential(nn.Linear(512+64,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,5))

    def forward(self, x, goal_ids):
        return self.fc(torch.cat([self.backbone(x).flatten(1),self.goal_embed(goal_ids)],dim=1))

def preprocess(frame):
    if frame.shape[0]!=224 or frame.shape[1]!=224:
        frame = cv2.resize(frame,(224,224))
    frame = frame.astype(np.float32)/255.0
    mean = np.array([0.485,0.456,0.406],dtype=np.float32)
    std = np.array([0.229,0.224,0.225],dtype=np.float32)
    return (frame-mean)/std.transpose(2,0,1)

def execute_action(pred, prev):
    if pred != prev:
        pyautogui.keyUp('w')
        if pred == 1:
            pyautogui.keyDown('w')
        elif pred == 2:
            pyautogui.move(-MOUSE_DELTA, 0)
        elif pred == 3:
            pyautogui.move(MOUSE_DELTA, 0)
        elif pred == 4:
            pyautogui.press('space')
    return pred

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V5Model(num_goals=args.num_goals).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print("[v5.1] Model loaded: " + args.model)
    print("[v5.1] Actions: " + str(ACTION_NAMES))
    print("[v5.1] Goal ID: " + str(args.goal_id) + ", Duration: " + str(args.duration) + "s")
    print("[v5.1] Frame stack: " + str(NUM_FRAMES) + " frames")
    print("[v5.1] Press Ctrl+C to stop")

    frame_window = []
    prev_action = 0

    for t in range(args.duration):
        frame = np.array(pyautogui.screenshot())[:,:,::-1]
        x = preprocess(frame)
        frame_window.append(x)
        if len(frame_window) > NUM_FRAMES:
            frame_window.pop(0)
        while len(frame_window) < NUM_FRAMES:
            frame_window.insert(0, frame_window[0])
        stacked = np.concatenate(frame_window, axis=0)
        x = torch.from_numpy(stacked).unsqueeze(0).float().to(device)
        gid = torch.tensor([args.goal_id], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x, gid)
            pred = torch.argmax(logits, dim=1).item()
        prev_action = execute_action(pred, prev_action)
        if pred != prev_action or t % 10 == 0:
            print("  [" + str(t) + "s] Action: " + ACTION_NAMES[pred])
        time.sleep(1.0)

    pyautogui.keyUp('w')

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--goal-id", type=int, default=0)
    p.add_argument("--num-goals", type=int, default=2)
    p.add_argument("--duration", type=int, default=60)
    a = p.parse_args()
    main(a)
