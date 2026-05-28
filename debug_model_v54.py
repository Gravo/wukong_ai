"""debug_model_v54.py - 诊断 v5.4 模型到底学没学会转向"""

import torch, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.goal_conditioned_bc_v54 import DualHeadModel  # 导入训练时的模型

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = DualHeadModel(num_actions=3, num_mouse_buckets=7).to(DEVICE)
ckpt = torch.load("checkpoints\\goal_bc_v54_best.pt", map_location=DEVICE)
model.load_state_dict(ckpt)
model.eval()
print("[诊断] 模型已加载\n")

# 测试1：随机噪声输入，看预测分布
print("=" * 60)
print("[测试1] 随机噪声输入 (100次) → 预测分布")
action_counts = {0:0, 1:0, 2:0}
bucket_counts = {i:0 for i in range(7)}

with torch.no_grad():
    for _ in range(100):
        fake_frames = torch.randn(1, 12, 224, 224).to(DEVICE)
        fake_goal = torch.tensor([0], dtype=torch.long).to(DEVICE)
        action_logits, mouse_logits = model(fake_frames, fake_goal)
        a = torch.argmax(action_logits, dim=1).item()
        b = torch.argmax(mouse_logits, dim=1).item()
        action_counts[a] += 1
        bucket_counts[b] += 1

print(f"  Action 分布: forward={action_counts[0]}, left={action_counts[1]}, right={action_counts[2]}")
print(f"  Bucket 分布: {[bucket_counts[i] for i in range(7)]}")
if action_counts[0] == 100:
    print("  ❌ 模型完全崩溃：只预测 forward！")
elif action_counts[0] > 80:
    print("  ⚠️  模型严重偏向 forward")
else:
    print("  ✅ 模型有多样性")

# 测试2：用真实数据评估
print("\n" + "=" * 60)
print("[测试2] 用真实数据评估 (pathfinding_data_noidle)")

import h5py, cv2

DATA_DIR = "pathfinding_data_noidle"
h5_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.h5')]
print(f"  找到 {len(h5_files)} 个文件")

# 只测前3个文件
total_samples = 0
correct_action = 0
correct_bucket = 0
pred_action_dist = {0:0, 1:0, 2:0}
true_action_dist = {0:0, 1:0, 2:0}

with torch.no_grad():
    for fname in h5_files[:3]:
        path = os.path.join(DATA_DIR, fname)
        with h5py.File(path, 'r') as f:
            frames = f['frames'][:]
            actions = f['actions'][:]
            mouse_dx = f['mouse_dx'][:]
            goal_ids = f['goal_ids'][:]
            
            # Bucket 计算（和训练时一样）
            def calc_bucket(dx):
                abs_dx = abs(dx)
                if abs_dx < 30: return 3
                elif abs_dx < 60: return 2 if dx < 0 else 4
                elif abs_dx < 150: return 1 if dx < 0 else 5
                else: return 0 if dx < 0 else 6
            
            n = min(50, len(frames) - 8)  # 最多测50帧
            for i in range(n):
                base = i
                stacked = []
                for offset in [0, 1, 3, 7]:
                    fi = base + offset
                    if fi < len(frames):
                        frame = frames[fi].astype(np.float32) / 255.0
                        frame = (frame - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
                        stacked.append(frame.transpose(2,0,1))
                    else:
                        stacked.append(np.zeros((3,224,224), dtype=np.float32))
                
                x = torch.from_numpy(np.concatenate(stacked, axis=0)).unsqueeze(0).float().to(DEVICE)
                gid = torch.tensor([goal_ids[base] % 2], dtype=torch.long).to(DEVICE)
                
                action_logits, mouse_logits = model(x, gid)
                pred_a = torch.argmax(action_logits, dim=1).item()
                pred_b = torch.argmax(mouse_logits, dim=1).item()
                
                # 真实标签
                true_a = int(actions[base + 1]) if base + 1 < len(actions) else 0
                true_dx = mouse_dx[base + 1] if base + 1 < len(mouse_dx) else 0
                true_b = calc_bucket(true_dx)
                
                pred_action_dist[pred_a] += 1
                true_action_dist[true_a] += 1
                total_samples += 1
                
                if pred_a == true_a:
                    correct_action += 1
                if pred_b == true_b:
                    correct_bucket += 1

print(f"  测试样本数: {total_samples}")
print(f"  Action Acc: {100.0*correct_action/total_samples:.1f}%")
print(f"  Bucket Acc: {100.0*correct_bucket/total_samples:.1f}%")
print(f"  真实 Action 分布: forward={true_action_dist[0]}, left={true_action_dist[1]}, right={true_action_dist[2]}")
print(f"  预测 Action 分布: forward={pred_action_dist[0]}, left={pred_action_dist[1]}, right={pred_action_dist[2]}")

if pred_action_dist[0] == total_samples:
    print("\n  ❌❌❌ 模型完全没学会转向！只预测 forward！")
    print("  原因：训练时 action_head 没被有效训练")
    print("  解决：需要重新训练，检查损失函数权重")

print("\n" + "=" * 60)
print("[诊断完成]")
