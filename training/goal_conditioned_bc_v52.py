"""goal_conditioned_bc_v52.py - v5.2 Non-uniform Frame Stacking + Mouse Speed + Focal Loss

Key fixes:
1. Causal alignment: actions[i] labels frames[i] (decision made WHEN seeing frame i)
2. Non-uniform stacking: [0, +1, +3, +7] covers 0/67ms/200ms/467ms
3. Mouse speed quantization: dx/dt -> idle/slow/medium/fast (30/150/500 px/s)
4. Conv1: pretrained copy + 0.01 perturbation for later frames
5. Zero-padding for boundary frames
6. Focal Loss (gamma=2.0)
7. Lazy loading: __getitem__ reads from disk (OOM-safe)
8. Memory-optimized: pre-computed normalization, no_grad accuracy, gc.collect
"""

import os, sys, time, argparse, h5py, glob, gc, math
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import ResNet18_Weights

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_FRAMES = 4
STACK_OFFSETS = [0, 1, 3, 7]
SPEED_THRESHOLDS = [30.0, 150.0, 500.0]
NUM_CLASSES = 5
ACTION_NAMES = ["idle", "forward", "turn_slow", "turn_medium", "turn_fast"]

# Pre-compute ImageNet normalization (avoid repeated allocation)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_ZERO_FRAME = np.zeros((224, 224, 3), dtype=np.float32)


def quantize_mouse_speed(mouse_dx, dt):
    if dt < 1e-6:
        return 0
    speed = abs(mouse_dx) / dt
    if speed < SPEED_THRESHOLDS[0]:
        return 0
    elif speed < SPEED_THRESHOLDS[1]:
        return 2
    elif speed < SPEED_THRESHOLDS[2]:
        return 3
    return 4


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


class V52Dataset(Dataset):
    """Lazy-loading: stores only metadata, reads frames on demand from disk."""

    def __init__(self, data_dir, max_samples=0):
        self.samples = []
        self.action_counts = [0] * NUM_CLASSES
        self.file_meta = {}
        self._build(data_dir)

        total = sum(self.action_counts)
        print(f"\n[Data] {len(self.samples)} samples:")
        for i in range(NUM_CLASSES):
            print(f"  {i}. {ACTION_NAMES[i]}: {self.action_counts[i]} ({100.0*self.action_counts[i]/max(total,1):.1f}%)")

        if 0 < max_samples < len(self.samples):
            idx = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in idx]
            print(f"[Data] Sampled -> {max_samples}")

    def _build(self, data_dir):
        max_offset = max(STACK_OFFSETS)
        print("[Data] Building index (lazy)...")
        for h5_path in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
            try:
                # Open, read small arrays, close immediately
                with h5py.File(h5_path, "r") as f:
                    if "frames" not in f:
                        continue
                    n = len(f["frames"])
                    fps = float(f.attrs.get("fps", 15.0))
                    has_mouse = bool(f.attrs.get("has_mouse", False))
                    actions = f["actions"][:]  # small: int array
                    goal_ids = f["goal_ids"][:] if "goal_ids" in f else np.zeros(n, dtype=np.int8)
                    mouse_dx = f["mouse_dx"][:] if (has_mouse and "mouse_dx" in f) else None

                self.file_meta[h5_path] = n
                print(f"  {os.path.basename(h5_path)}: {n}f mouse={has_mouse}")

                dt = 1.0 / fps
                max_idx = n - max_offset - 1
                for i in range(1, max_idx + 1):
                    raw = int(actions[i])
                    if has_mouse and mouse_dx is not None:
                        dx = float(mouse_dx[i])
                        sc = quantize_mouse_speed(dx, dt)
                        ac = 1 if (raw == 4 and sc == 0) else (0 if sc == 0 else sc)
                    else:
                        ac = 1 if raw == 4 else (0 if raw == 0 else 2)
                    self.samples.append((h5_path, i, ac, int(goal_ids[i])))
                    self.action_counts[ac] += 1

                # Free numpy arrays immediately
                del actions, goal_ids
                if mouse_dx is not None:
                    del mouse_dx
                gc.collect()
            except Exception as e:
                print(f"  Error: {os.path.basename(h5_path)}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_path, base_idx, action_class, goal_id = self.samples[idx]
        n = self.file_meta[h5_path]

        stacked = []
        with h5py.File(h5_path, "r") as f:
            frames_ds = f["frames"]
            for offset in STACK_OFFSETS:
                fi = base_idx + offset
                if 0 <= fi < n:
                    # Read + normalize in one step, avoid extra array copies
                    frame = frames_ds[fi]  # returns numpy array from h5
                    frame = (frame.astype(np.float32) / 255.0 - _MEAN) / _STD
                else:
                    frame = (_ZERO_FRAME - _MEAN) / _STD  # pre-normalized zero
                stacked.append(frame.transpose(2, 0, 1))

        # Single concatenation instead of list of arrays
        return (
            torch.from_numpy(np.concatenate(stacked)).float(),
            torch.tensor(action_class, dtype=torch.long),
            torch.tensor(goal_id, dtype=torch.long),
        )


class V52Model(nn.Module):
    def __init__(self, num_goals, num_frames=NUM_FRAMES, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        old = resnet.conv1
        new = nn.Conv2d(3 * num_frames, 64, 7, stride=2, padding=3, bias=False)
        if pretrained and old.weight is not None:
            with torch.no_grad():
                # Scale by 1/num_frames to maintain output variance
                # when stacking multiple frames (12 input channels vs 3 original)
                for c in range(num_frames):
                    new.weight[:, c*3:(c+1)*3].copy_(old.weight * (1.0 / num_frames))
        resnet.conv1 = new
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.goal_embed = nn.Embedding(num_goals, 64)
        self.fc = nn.Sequential(nn.Linear(512+64, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))

    def forward(self, x, goal_ids):
        return self.fc(torch.cat([self.backbone(x).flatten(1), self.goal_embed(goal_ids)], 1))


def train_epoch(model, loader, opt, device, epoch, fl_fn):
    model.train()
    total_loss = correct = total = 0
    for bi, (frames, actions, goal_ids) in enumerate(loader):
        frames, actions, goal_ids = frames.to(device), actions.to(device), goal_ids.to(device)
        opt.zero_grad(set_to_none=True)

        logits = model(frames, goal_ids)
        loss = fl_fn(logits, actions)
        loss.backward()

        # Accuracy under no_grad (key fix: avoids 2nd forward with grad graph)
        with torch.no_grad():
            _, pred = logits.argmax(dim=1)
            correct += (pred == actions).sum().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item()
        total += actions.size(0)

        # Free references
        del logits, loss
        if bi % 50 == 0:
            print(f"  E{epoch} B{bi}/{len(loader)} Loss:{total_loss/(bi+1):.4f}")

        # Periodic GC to release h5py temp buffers
        if bi % 200 == 0 and bi > 0:
            gc.collect()

    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  v5.2 Non-uniform Stack + Mouse Speed + Focal Loss")
    print(f"  Offsets: {STACK_OFFSETS} | Speed: {SPEED_THRESHOLDS} px/s")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    ds = V52Dataset(args.data_dir, args.max_samples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    num_goals = max((s[3] for s in ds.samples), default=0) + 1
    print(f"[Train] {num_goals} goals, {len(ds)} samples, bs={args.batch_size}")

    model = V52Model(num_goals).to(device)

    counts = ds.action_counts
    total_c = max(sum(counts), 1)
    alpha = torch.tensor([math.sqrt(total_c / max(c, 1)) for c in counts], dtype=torch.float32).to(device)
    print(f"[Train] Focal weights: {[f'{w:.2f}' for w in alpha.tolist()]}")
    fl_fn = FocalLoss(alpha=alpha, gamma=2.0)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_acc = 0.0

    for ep in range(1, args.epochs + 1):
        gc.collect()
        t0 = time.time()
        loss, acc = train_epoch(model, dl, opt, device, ep, fl_fn)
        sched.step()
        dt = time.time() - t0
        if acc > best_acc:
            best_acc = acc
        print(f"\n  Epoch {ep}/{args.epochs}  Loss:{loss:.4f}  Acc:{acc:.2f}%  Best:{best_acc:.2f}%  Time:{dt:.1f}s\n")
        if ep % args.save_interval == 0 or ep == args.epochs:
            ckpt = os.path.join(args.output_dir, f"goal_bc_v52_epoch_{ep:03d}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  [Save] {ckpt}")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "goal_bc_v52_final.pt"))
    print(f"\n[Done] Best Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--output-dir", default="checkpoints")
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    main(a)
