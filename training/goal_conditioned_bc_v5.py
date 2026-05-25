import os, sys, time, argparse, h5py, glob, json, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===== 动作空间定义 =====
ACTION_MAP = {0:0, 4:1, 6:2, 5:3, 3:4}
ACTION_NAMES = ["idle","forward","turn_left","turn_right","dodge"]
NUM_CLASSES = 5

class V5Dataset(Dataset):
    def __init__(self, data_dir, max_samples=0):
        self.samples = []
        self.action_counts = [0]*NUM_CLASSES
        print(f"[Data] Loading from {data_dir}...")
        for h5_file in sorted(glob.glob(os.path.join(data_dir,"*.h5"))):
            self._load_file(h5_file)
        total = sum(self.action_counts)
        print(f"[Data] Loaded {len(self.samples)} samples:")
        for i,n in enumerate(ACTION_NAMES):
            print(f"  {i}. {n}: {self.action_counts[i]} ({100.0*self.action_counts[i]/max(total,1):.1f}%)")
        if max_samples > 0 and len(self.samples) > max_samples:
            idx = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in idx]

    def _load_file(self, h5_path):
        try:
            with h5py.File(h5_path,'r') as f:
                if 'frames' not in f: return
                frames=f['frames'][:]; actions=f['actions'][:]
                goal_ids=f['goal_ids'][:] if 'goal_ids' in f else np.zeros(len(frames),dtype=np.int8)
                for i in range(len(frames)):
                    raw=int(actions[i])
                    if raw not in ACTION_MAP: continue
                    v5=ACTION_MAP[raw]
                    self.samples.append((h5_path,i,v5,int(goal_ids[i])))
                    self.action_counts[v5]+=1
        except Exception as e:
            print(f"[Data] Error: {e}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        h5_path,fidx,action,goal_id=self.samples[idx]
        with h5py.File(h5_path,'r') as f:
            frame=f['frames'][fidx]
        if frame.shape[0]!=224 or frame.shape[1]!=224:
            frame=cv2.resize(frame,(224,224))
        frame=frame.astype(np.float32)/255.0
        mean=np.array([0.485,0.456,0.406],dtype=np.float32).reshape(1,1,3)
        std=np.array([0.229,0.224,0.225],dtype=np.float32).reshape(1,1,3)
        frame=(frame-mean)/std
        frame=frame.transpose(2,0,1)
        return torch.from_numpy(frame),torch.tensor(action,dtype=torch.long),torch.tensor(goal_id,dtype=torch.long)

class V5Model(nn.Module):
    def __init__(self, num_goals, pretrained=True):
        super().__init__()
        resnet=models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone=nn.Sequential(*list(resnet.children())[:-1])
        self.goal_embed=nn.Embedding(num_goals,64)
        self.fc=nn.Sequential(nn.Linear(512+64,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,NUM_CLASSES))

    def forward(self, x, goal_ids):
        feat=self.backbone(x).flatten(1)
        g=self.goal_embed(goal_ids)
        return self.fc(torch.cat([feat,g],dim=1))

def train_epoch(model, loader, optim, device, epoch, weights):
    model.train()
    criterion=nn.CrossEntropyLoss(weight=weights.to(device))
    total_loss=correct=total=0
    for bi,(frames,actions,goal_ids) in enumerate(loader):
        frames,actions,goal_ids=frames.to(device),actions.to(device),goal_ids.to(device)
        optim.zero_grad()
        loss=criterion(model(frames,goal_ids),actions)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0); optim.step()
        total_loss+=loss.item()
        _,p=torch.max(model(frames,goal_ids),1)
        correct+=(p==actions).sum().item(); total+=actions.size(0)
        if bi%100==0: print(f"  E{epoch}B{bi}/{len(loader)} Loss:{loss.item():.4f}",flush=True)
    return total_loss/max(len(loader),1),100.0*correct/max(total,1)

def main(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\n  v5 Discrete Actions\n  Device: {device}\n{'='*60}\n")
    ds=V5Dataset(args.data_dir,args.max_samples)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    ng=max((s[3] for s in ds.samples),default=0)+1
    print(f"[Train] {ng} goals, {len(ds)} samples")
    model=V5Model(ng).to(device)
    opt=optim.Adam(model.parameters(),lr=args.lr)
    counts=ds.action_counts; total=sum(counts)
    w=[total/(NUM_CLASSES*max(c,1))*(5 if i>0 else 1) for i,c in enumerate(counts)]
    weights=torch.tensor([x/sum(w)*NUM_CLASSES for x in w],dtype=torch.float32)
    print(f"[Train] Weights: {[f'{x:.2f}' for x in weights.tolist()]}")
    for ep in range(1,args.epochs+1):
        t0=time.time()
        loss,acc=train_epoch(model,dl,opt,device,ep,weights)
        dt=time.time()-t0
        print(f"\n  Epoch {ep}/{args.epochs} Loss:{loss:.4f} Acc:{acc:.2f}% Time:{dt:.1f}s\n")
        if ep%args.save_interval==0 or ep==args.epochs:
            p=os.path.join(args.output_dir,f"goal_bc_v5_epoch_{ep:03d}.pt")
            torch.save(model.state_dict(),p); print(f"  [Save] {p}")
    torch.save(model.state_dict(),os.path.join(args.output_dir,"goal_bc_v5_final.pt"))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",required=True); p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--lr",type=float,default=0.001)
    p.add_argument("--max-samples",type=int,default=0); p.add_argument("--save-interval",type=int,default=10)
    p.add_argument("--output-dir",default="checkpoints")
    a=p.parse_args(); os.makedirs(a.output_dir,exist_ok=True); main(a)
