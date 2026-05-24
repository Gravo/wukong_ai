import os, sys, argparse, numpy as np, torch, torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import pyautogui, time, cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ACTION_NAMES=["idle","forward","turn_left","turn_right","dodge"]
MOUSE_DELTA=86

class V5Model(nn.Module):
    def __init__(self,num_goals,pretrained=True):
        super().__init__()
        resnet=models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone=nn.Sequential(*list(resnet.children())[:-1])
        self.goal_embed=nn.Embedding(num_goals,64)
        self.fc=nn.Sequential(nn.Linear(512+64,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,5))

    def forward(self,x,goal_ids):
        return self.fc(torch.cat([self.backbone(x).flatten(1),self.goal_embed(goal_ids)],dim=1))

def preprocess(frame):
    if frame.shape[0]!=224 or frame.shape[1]!=224:
        frame=cv2.resize(frame,(224,224))
    frame=frame.astype(np.float32)/255.0
    mean=np.array([0.485,0.456,0.406],dtype=np.float32).reshape(1,1,3)
    std=np.array([0.229,0.224,0.225],dtype=np.float32).reshape(1,1,3)
    return (frame-mean)/std.transpose(2,0,1)

def execute_action(pred):
    if pred==0: pass
    elif pred==1: pyautogui.keyDown('w')
    elif pred==2: pyautogui.move(-MOUSE_DELTA,0)
    elif pred==3: pyautogui.move(MOUSE_DELTA,0)
    elif pred==4: pyautogui.press('space')

def main(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=V5Model(num_goals=2).to(device)
    model.load_state_dict(torch.load(args.model,map_location=device))
    model.eval()
    print(f"[v5] Model loaded: {args.model}")
    print(f"[v5] Actions: {ACTION_NAMES}")
    print(f"[v5] Goal ID: {args.goal_id}, Duration: {args.duration}s")
    print(f"[v5] Press Ctrl+C to stop")
    prev=0
    for t in range(args.duration):
        frame=np.array(pyautogui.screenshot())[:,:,::-1]
        x=torch.from_numpy(preprocess(frame)).unsqueeze(0).float().to(device)
        gid=torch.tensor([args.goal_id],dtype=torch.long).to(device)
        with torch.no_grad():
            logits=model(x,gid)
            pred=torch.argmax(logits,dim=1).item()
        if pred!=prev:
            execute_action(pred)
            print(f"  [{t}s] Action: {ACTION_NAMES[pred]}")
            prev=pred
        time.sleep(1.0)
    pyautogui.keyUp('w')

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True); p.add_argument("--goal-id",type=int,default=0)
    p.add_argument("--duration",type=int,default=60)
    a=p.parse_args(); main(a)
