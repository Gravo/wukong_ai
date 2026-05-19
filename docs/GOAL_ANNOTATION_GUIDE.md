# Goal-Conditioned BC 使用指南

**目标**：解决「模型不知道目标」的根本问题，实现可用的寻路AI

---

## 🚀 快速开始（4步完成第一次训练）

### 第1步：定义目标序列

创建 `goals.txt`（每行一个目标，按游戏路径顺序）：

```txt
存档点A
大树守卫战
岔路口（左去BOSS/右去道具）
BOSS门
存档点B
```

**示例**（黑神话悟空-第一章）：
```txt
存档点A
第一个小BOSS
岔路口
BOSS门
存档点B
```

### 第2步：录制带 Goal 标注的数据

```powershell
cd D:\projects\wukong_ai

C:\Python\python.exe -u training\data_collector_v3.py ^
  --duration 600 ^
  --fps 15 ^
  --goals-file goals.txt ^
  --output pathfinding_data_goal
```

**游戏中操作**：
1. 开始录制后，正常游玩
2. **到达一个目标后，按 `G` 键标记**（脚本自动切换到下一个目标）
3. 继续走向下一个目标
4. 所有目标都到达后，按 `ESC` 停止录制

**重要**：
- 按 `G` 键标记「到达当前目标」
- 脚本会在终端显示当前目标（例如：`Goal: 岔路口 (goal_id=2) | Press 'G' to advance`）
- 录制结束后会显示 Goal 分布（每个目标有多少帧）

### 第3步：检查录制质量

```powershell
C:\Python\python.exe -u training\data_collector_v3.py --report pathfinding_data_goal
```

**合格标准**（终端会显示）：
- ✅ 数据量: ≥3000帧
- ✅ idle占比: <70%
- ✅ 动作多样性: ≥3种非idle动作
- ✅ 鼠标活跃度: >5%
- ✅ 所有目标都已到达

如果不合格，重新录制（调整 gameplay）。

### 第4步：训练 Goal-Conditioned BC 模型

```powershell
cd D:\projects\wukong_ai

C:\Python\python.exe -u training\goal_conditioned_bc.py ^
  --data-dir pathfinding_data_goal ^
  --epochs 50 ^
  --batch-size 32 ^
  --lr 0.001 ^
  --use-lstm
```

**训练过程**：
- 每10个batch显示一次损失
- 每个epoch结束后显示损失、准确率、耗时
- 每10个epoch保存一次检查点（`checkpoints/goal_bc_epoch_XXX.pt`）
- 训练完成后，最佳模型在 `checkpoints/goal_bc_epoch_050.pt`

### 第5步：推理测试

```powershell
C:\Python\python.exe -u training\inference_goal.py ^
  --model checkpoints\goal_bc_epoch_050.pt ^
  --duration 60 ^
  --fps 10
```

**观察**：
- AI 是否能够「寻路」（走向目标）？
- 动作是否多样化（不再只输出 forward）？
- 鼠标控制是否正常？

---

## 📋 详细文档

### 1. Goal 标注原理

**问题**：之前的 BC 模型失败，因为「不知道目标在哪」。

**解决**：给每帧标注 `goal_id`（当前目标），让模型「知道要去哪」。

**实现**：
- 录制前定义目标序列（例如：存档点A → 岔路口 → BOSS门）
- 录制中按 `G` 键标记「到达当前目标」
- 脚本自动给每帧标注 `goal_id`
- 训练时，模型同时看到「画面」和「当前目标」，学会「朝着目标走」

### 2. 如何定义 Goal（目标）？

**原则**：
1. **按游戏路径顺序**（从起点到终点）
2. **每个目标对应一个「里程碑」**（存档点、BOSS门、岔路口等）
3. **目标数量 3-7 个**（太少=没帮助，太多=每个目标数据太少）

**示例**（黑神话悟空-第一章）：
```txt
# goals.txt
存档点A
大树守卫战
岔路口（左去BOSS/右去道具）
BOSS门
存档点B
```

**高级**：如果你有多个路径（例如：去BOSS vs 去道具），可以录多个 episode，每个 episode 有不同的 goal 序列。

### 3. 录制时的技巧

**DO**：
- ✅ 按 `G` 键要及时（到达目标后立即按）
- ✅ 每个目标之间要走不同的路线（增加多样性）
- ✅ 录制多个 episode（至少 3-5 个）
- ✅ 检查终端输出的 Goal 分布（每个目标应该有足够多的帧）

**DON'T**：
- ❌ 忘记按 `G` 键（会导致 goal_id 错误）
- ❌ 一个目标停留太久（会导致数据不平衡）
- ❌ 只录一个 episode（数据量太少）

### 4. 训练参数调优

**基础参数**（第一次训练用这些）：
```powershell
--epochs 50 ^
--batch-size 32 ^
--lr 0.001 ^
--use-lstm
```

**如果效果不好，尝试**：
1. **降低学习率**（`--lr 0.0001`）
2. **增加 batch size**（`--batch-size 64`，如果显存够）
3. **增加 epoch 数量**（`--epochs 100`）
4. **去掉 LSTM**（`--use-lstm` 去掉，如果序列建模没帮助）

**如果过拟合**（训练准确率很高，但推理效果差）：
1. **增加 dropout**（`--dropout 0.7`）
2. **减少 LSTM 层数**（`--lstm-layers 1`）
3. **减少训练数据**（只用一部分数据训练）

### 5. 评估模型效果

**量化指标**：
- **成功率**：AI 能够到达目标的百分比（≥50% 算合格）
- **平均用时**：AI 用时 / 人类用时（≤1.5× 算合格）
- **动作多样性**：AI 是否输出多种动作（不再只输出 forward）

**可视化分析**：
- 录制推理过程的视频（用 OBS 或其他录屏软件）
- 观察 AI 是否能够「寻路」
- 观察 AI 在岔路口是否能够正确选择

**如果效果不好**：
1. **检查数据质量**（是否有「坏帧」？是否按 `G` 键及时？）
2. **增加数据量**（录制更多 episode）
3. **调整模型架构**（增加/减少 LSTM 层数、隐藏维度等）
4. **尝试其他方案**（Decision Transformer、DAgger 等）

---

## 🔬 高级话题

### A. 如何自动发现 Goal（无需人工标注）？

**方法**：对成功轨迹做 K-Means 聚类（k=5），聚类中心 = 自动发现的子目标。

**步骤**：
1. 训练一个「特征提取器」（ResNet18）
2. 提取所有成功轨迹的帧特征（256维）
3. K-Means 聚类（k=5）
4. 聚类中心 = 自动发现的子目标
5. 推理时：当前帧 → 找最近的聚类中心 → 作为 goal_id

**代码**（简化版）：
```python
from sklearn.cluster import KMeans

# 1. 提取特征
features = []
for traj in successful_trajectories:
    for frame in traj['frames']:
        feat = resnet(preprocess(frame))  # [256]
        features.append(feat.detach().numpy())

# 2. 聚类
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(features)
cluster_centers = kmeans.cluster_centers_  # [5, 256]

# 3. 推理时使用
def get_goal_id(current_frame):
    feat = resnet(preprocess(current_frame))  # [256]
    distances = [np.linalg.norm(feat - center) for center in cluster_centers]
    return np.argmin(distances)
```

### B. 如何添加「自适应计算」（老花眼开车）？

**洞察**：大部分帧不需要精细处理，只有分叉路才需要「仔细看」。

**实现**：
1. **关键帧检测**（光流/特征变化）
2. **级联模型**（小模型快速过滤 + 大模型精细决策）

**代码**（简化版）：
```python
class AdaptiveWukongAI:
    def __init__(self, small_model, large_model):
        self.small_model = small_model  # MobileNetV2 (快速）
        self.large_model = large_model  # VideoMAE-Small (精确）
        self.keyframe_detector = OpticalFlowKeyFrameDetector()
    
    def predict(self, frame):
        if self.keyframe_detector.is_keyframe(frame):
            print("关键帧：使用大模型")
            return self.large_model(frame)
        else:
            print("普通帧：使用小模型")
            return self.small_model(frame)
```

**但是**：自适应计算是「锦上添花」，先解决根本问题（添加 goal 变量）。

### C. 如何升级到 Decision Transformer？

**如果 Goal-Conditioned BC 效果不够好**，可以尝试 Decision Transformer。

**Decision Transformer 优势**：
- 专为「序列决策」设计（完美匹配 wukong_ai）
- 可以很小（2M 参数，RTX 2060 轻松跑）
- 离线训练（不需要在线交互）
- 时序建模能力强（Transformer）

**但是**：
- 需要自己实现（无现成代码）
- 需要设计回报函数（reward function）
- 风险更高（可能实现有误）

**预估时间**：5-7天（如果 Goal-Conditioned BC 失败）

---

## 📁 文件清单

| 文件 | 作用 | 使用方法 |
|------|------|----------|
| `training\data_collector_v3.py` | 录制带 Goal 标注的数据 | `C:\Python\python.exe -u training\data_collector_v3.py --goals-file goals.txt` |
| `training\goal_conditioned_bc.py` | 训练 Goal-Conditioned BC 模型 | `C:\Python\python.exe -u training\goal_conditioned_bc.py --data-dir pathfinding_data_goal` |
| `training\inference_goal.py` | 推理测试（待实现） | `C:\Python\python.exe -u training\inference_goal.py --model checkpoints\goal_bc_epoch_050.pt` |
| `goals.txt` | 目标序列定义 | 每行一个目标（按游戏路径顺序） |

---

## 🎯 成功标准

### 最低标准（必须达到）
- [ ] 模型能「到达目标」（成功率 > 50%）
- [ ] 不再只输出 forward（动作分布多样化）

### 中等标准（期望达到）
- [ ] 成功率 > 80%
- [ ] 平均用时 ≤ 1.5×人类时间

### 最高标准（理想情况）
- [ ] 成功率 > 95%
- [ ] 平均用时 ≤ 1.2×人类时间
- [ ] 能处理未见过的场景（泛化能力）

---

## 📞 问题排查

### 问题1：按 `G` 键没反应？

**可能原因**：
- keyboard 库没有正确安装（`pip install keyboard`）
- 没有以管理员权限运行
- 游戏窗口没有焦点

**解决方法**：
1. 以管理员权限运行 PowerShell
2. 确保游戏窗口有焦点（点击游戏窗口）
3. 尝试按 `G` 键多次

### 问题2：训练时 loss 不下降？

**可能原因**：
- 数据量太少（<3000帧）
- 学习率太高（尝试 `--lr 0.0001`）
- 模型架构不对（尝试去掉 `--use-lstm`）

**解决方法**：
1. 增加数据量（录制更多 episode）
2. 降低学习率
3. 简化模型架构

### 问题3：推理时 AI 仍然只会往前走？

**可能原因**：
- Goal 标注错误（没有正确按 `G` 键）
- 数据不平衡（某个 goal 的帧太多/太少）
- 模型没有学到 goal 语义

**解决方法**：
1. 检查 Goal 分布（运行 `--report`）
2. 重新录制数据（确保正确按 `G` 键）
3. 增加数据多样性（走不同的路线）

---

## 🔗 参考文献

1. **Goal-Conditioned Imitation Learning**: "Learning Goal-Conditioned Policies from Current Behavior" (2020)
2. **Decision Transformer**: "Decision Transformer: Reinforcement Learning via Sequence Modeling" (2021)
3. **Sub-goal Discovery**: "Learning Sub-goals as Abstract Actions for Hierarchical Imitation Learning" (2020)

---

**文档版本**: v1.0 (2026-05-19)  
**作者**: QClaw (Top-level AI Researcher)  
**项目**: [wukong_ai](https://github.com/Gravo/wukong_ai)
