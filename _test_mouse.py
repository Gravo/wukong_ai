import sys, glob, torch, numpy as np, h5py
sys.path.insert(0, '.')
from pathfinding.behavior_clone_v2 import BehaviorCloneModel, H5Dataset

device = "cuda"
ckpt = torch.load("checkpoints/bc_best.pt", map_location=device)
model = BehaviorCloneModel(num_actions=10)
model.load_state_dict(ckpt)
model.to(device)
model.eval()
print("Model loaded")

h5_files = sorted(glob.glob("pathfinding_data/*.h5"))
print(f"Found {len(h5_files)} h5 files")
dataset = H5Dataset(h5_files[0], augment=False)
print(f"Dataset length: {len(dataset)}")

dxs, dys, acts = [], [], []
names = ["idle","atk","heavy","dodge","fwd","rgt","lft","d_atk","lock","heal"]
for i in range(min(50, len(dataset))):
    frames, action_gt, mouse_tgt = dataset[i]
    frames_t = frames.unsqueeze(0).to(device)
    with torch.no_grad():
        logits, mouse = model(frames_t)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        dx = mouse[0,0].item()
        dy = mouse[0,1].item()
        dxs.append(dx)
        dys.append(dy)
        acts.append(pred)
        if i < 5:
            print(f"  sample {i}: pred={pred}({names[pred]:6s}), dx={dx:.4f}, dy={dy:.4f}, gt_dx={mouse_tgt[0]:.4f}")

print()
print("=== Mouse Output on Real Frames ===")
print(f"dx: mean={np.mean(dxs):.4f}, std={np.std(dxs):.4f}, min={np.min(dxs):.4f}, max={np.max(dxs):.4f}")
print(f"dy: mean={np.mean(dys):.4f}, std={np.std(dys):.4f}, min={np.min(dys):.4f}, max={np.max(dys):.4f}")
print(f"Non-zero dx(>0.01): {sum(1 for d in dxs if abs(d)>0.01)}/50")
print(f"Non-zero dy(>0.01): {sum(1 for d in dys if abs(d)>0.01)}/50")
print(f"Action distribution: {dict(zip(*np.unique(acts, return_counts=True)))}")
