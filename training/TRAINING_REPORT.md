# Training Report - Goal-Conditioned BC Model
Date: 2026-05-23
Model: Goal-Conditioned Behavior Cloning (ResNet18 + Goal Embedding)

## Training Configuration
- Data: 18 files, 15,247 samples
- Epochs: 15
- Batch size: 32
- Learning rate: 0.001
- Device: CUDA (RTX 2060 4.5GB)
- Loss: action_CE + 2.0 * mouse_SmoothL1
- Goal embedding dim: 64

## Training Results

| Epoch | Loss   | Accuracy | Time (s) |
|-------|--------|----------|----------|
| 1     | 0.9150 | 65.55%   | 580.3    |
| 2     | 0.8113 | 69.78%   | 136.2    |
| 3     | 0.7482 | 72.43%   | 132.1    |
| 4     | 0.6887 | 75.14%   | 134.5    |
| 5     | 0.6196 | 77.36%   | 136.6    |
| 6     | 0.5325 | 80.79%   | 135.7    |
| 7     | 0.4309 | 84.83%   | 135.2    |
| 8     | 0.3214 | 88.75%   | 131.3    |
| 9     | 0.2286 | 92.50%   | 141.1    |
| 10    | 0.1490 | 95.63%   | 139.3    |
| 11    | 0.0994 | 97.42%   | 142.1    |
| 12    | 0.0612 | 98.62%   | 139.7    |
| 13    | 0.0399 | 99.13%   | 143.8    |
| 14    | 0.0321 | 99.44%   | 138.8    |
| 15    | 0.0290 | 99.53%   | 131.9    |

**Total training time: ~33 minutes**

## Model Analysis

### Prediction Distribution (first 1024 samples)
| Action  | Pred# | Target# | Accuracy | Data % |
|----------|-------|---------|----------|--------|
| idle     | 367   | 367     | 100.0%   | 35.8%  |
| forward  | 566   | 566     | 100.0%   | 55.3%  |
| right    | 74    | 74      | 100.0%   | 7.2%   |
| left     | 17    | 17      | 100.0%   | 1.7%   |
| others   | 0     | 0       | N/A      | 0.0%   |

### Key Findings
1. **Perfect overfitting**: 99.53% training accuracy (no validation set)
2. **Severe data imbalance**: Only 4 actions in training data
3. **No combat actions**: attack, dodge, lock, heal not in pathfinding data
4. **Model type**: This is a PATHFINDING model only, not a complete Wukong AI

## Checkpoints Saved
- `checkpoints/goal_bc_epoch_010.pt` (43.9 MB)
- `checkpoints/goal_bc_epoch_015.pt` (43.9 MB)

## Next Steps
1. Test inference in-game (need `inference_goal.py`)
2. Collect combat training data (attack, dodge, lock, heal)
3. Consider DAgger for online correction
4. Add validation set to detect overfitting

## Data Quality Issues
- 11 out of 17 files recorded on 5/23 have 0% mouse activity (pynput permissions)
- Only 2 files have good mouse data (ep1_1779502795, ep1_1779502885)
- Recommend re-recording with administrator privileges

---
*Report generated automatically on 2026-05-23 11:45 GMT+8*