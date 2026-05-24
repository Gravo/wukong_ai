"""goal_conditioned_bc_v51.py - Frame Stacking v5.1
堆叠4帧连续输入，让模型看到运动趋势"""

import os, sys, time, argparse, h5py, glob, cv2
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import ResNet18_Weights

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_FRAMES = 4
ACTION_MAP = {0:0, 4:1, 6:2, 5:3, 3:4}
ACTION_NAMES = ["idle","forward","turn_left","turn_right","dodge"]
NUM_CLASSES = 5

class V5Dataset(Dataset):
    def __init__(self, data_dir, max_samples=0):
        self.samples = []
        self.action_counts = [0]*NUM_CLASSES
        self.frame_history = {}
        self._load_all_frames(data_dir)
        self._build_samples()
        total = sum(self.action_counts)
        print("[Data] Loaded " + str(len(self.samples)) + " samples:")
        for i,n in enumerate(ACTION_NAMES):
            pct = 100.0*self.action_counts[i]/max(total,1)
            print("  " + str(i) + ". " + n + ": " + str(self.action_counts[i]) + " (" + "{:.1f}".format(pct) + "%)")
        if max_samples > 0 and len(self.samples) > max_samples:
            idx = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in idx]

    def _load_all_frames(self, data_dir):
        print("[Data] Loading frames...")
        for h5_file in sorted(glob.glob(os.path.join(data_dir,"*.h5"))):
            try:
                with h5py.File(h5_file,"r") as f:
                    if "frames" not in f: continue
                    frames = f["frames"][:].astype(np.uint8)
                    print("  " + h5_file + ": " + str(len(frames)) + " frames")
                    self.frame_history[h5_file] = frames
            except Exception as e:
                print("  Error: " + str(e))

    def _build_samples(self):
        for h5_path, frames in self.frame_history.items():
            with h5py.File(h5_path,"r") as f:
                actions = f["actions"][:]
                goal_ids = f["goal_ids"][:] if "goal_ids" in f else np.zeros(len(frames),dtype=np.int8)
            for i in range(NUM_FRAMES-1, len(frames)):
                raw = int(actions[i])
                if raw not in ACTION_MAP: continue
                v5 = ACTION_MAP[raw]
                self.samples.append((h5_path, i, v5, int(goal_ids[i])))
                self.action_counts[v5] += 1

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        h5_path, fidx, action, goal_id = self.samples[idx]
        frames = self.frame_history[h5_path]
        stacked = []
        for j in range(fidx-NUM_FRAMES+1, fidx+1):
            j = max(0, j)
            frame = frames[j]
            if frame.shape[0]!=224 or frame.shape[1]!=224:
                frame = cv2.resize(frame,(224,224))
            frame = frame.astype(np.float32)/255.0
            mean = np.array([0.485,0.456,0.406],dtype=np.float32)
            std = np.array([0.229,0.224,0.225],dtype=np.float32)
            frame = (frame-mean)/std
            stacked.append(frame.transpose(2,0,1))
        stacked = np.concatenate(stacked, axis=0)
        return torch.from_numpy(stacked), torch.tensor(action,dtype=torch.long), torch.tensor(goal_id,dtype=torch.long)

class V5Model(nn.Module):
    def __init__(self, num_goals, num_frames=NUM_FRAMES, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(3*num_frames, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight[:,:3,:,:] = old_conv.weight
            for c in range(3, 3*num_frames):
                resnet.conv1.weight[:,c,:,:] = old_conv.weight[:,c%3,:,:]
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.goal_embed = nn.Embedding(num_goals, 64)
        self.fc = nn.Sequential(nn.Linear(512+64,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,NUM_CLASSES))

    def forward(self, x, goal_ids):
        feat = self.backbone(x).flatten(1)
        return self.fc(torch.cat([feat,self.goal_embed(goal_ids)],dim=1))

def train_epoch(model, loader, opt, device, epoch, weights):
    model.train()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    total_loss = correct = total = 0
    for bi,(frames,actions,goal_ids) in enumerate(loader):
        frames,actions,goal_ids = frames.to(device),actions.to(device),goal_ids.to(device)
        opt.zero_grad()
        loss = criterion(model(frames,goal_ids), actions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        opt.step()
        total_loss += loss.item()
        _,p = torch.max(model(frames,goal_ids),1)
        correct += (p==actions).sum().item()
        total += actions.size(0)
        if bi%100==0:
            print("  E" + str(epoch) + "B" + str(bi) + "/" + str(len(loader)) + " Loss:" + "{:.4f}".format(loss.item()))
    return total_loss/max(len(loader),1), 100.0*correct/max(total,1)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("")
    print("="*60)
    print("  v5.1 Frame Stacking (" + str(NUM_FRAMES) + " frames)")
    print("  Device: " + str(device))
    print("="*60)
    print("")
    ds = V5Dataset(args.data_dir, args.max_samples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    ng = max((s[3] for s in ds.samples),default=0)+1
    print("[Train] " + str(ng) + " goals, " + str(len(ds)) + " samples, " + str(NUM_FRAMES) + " frames stacked")
    model = V5Model(ng).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    counts = ds.action_counts
    total = sum(counts)
    w = [total/(NUM_CLASSES*max(c,1))*(5 if i>0 else 1) for i,c in enumerate(counts)]
    weights = torch.tensor([x/sum(w)*NUM_CLASSES for x in w], dtype=torch.float32)
    print("[Train] Weights: " + str(["{:.2f}".format(x) for x in weights.tolist()]))
    for ep in range(1,args.epochs+1):
        t0 = time.time()
        loss,acc = train_epoch(model,dl,opt,device,ep,weights)
        dt = time.time()-t0
        print("")
        print("  Epoch " + str(ep) + "/" + str(args.epochs) + " Loss:" + "{:.4f}".format(loss) + " Acc:" + "{:.2f}".format(acc) + "% Time:" + "{:.1f}".format(dt) + "s")
        print("")
        if ep%args.save_interval==0 or ep==args.epochs:
            p = os.path.join(args.output_dir,"goal_bc_v51_epoch_" + "{:03d}".format(ep) + ".pt")
            torch.save(model.state_dict(),p)
            print("  [Save] " + p)
    torch.save(model.state_dict(), os.path.join(args.output_dir,"goal_bc_v51_final.pt"))

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",required=True)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--lr",type=float,default=0.001)
    p.add_argument("--max-samples",type=int,default=0)
    p.add_argument("--save-interval",type=int,default=10)
    p.add_argument("--output-dir",default="checkpoints")
    a = p.parse_args()
    os.makedirs(a.output_dir,exist_ok=True)
    main(a)
