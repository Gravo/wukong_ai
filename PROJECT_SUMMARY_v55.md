# wukong_ai 项目总结 - v5.5 训练完成版

**最后更新**: 2026-05-28 21:55  
**状态**: ✅ v5.5 训练完成，推理脚本已修复，待游戏内测试

---

## 项目概述

**目标**: 训练 AI 让《黑神话：悟空》角色自主导航（寻路+转向）

**方法**: Goal-Conditioned Behavior Cloning（目标条件化行为克隆）

**数据集**: `pathfinding_data_balanced/` (9,790 样本，Bucket 3 占比从 81.15% 降至 46.27%)

---

## 训练结果（v5.5）

| 指标 | 值 | 评价 |
|------|-----|------|
| **Val Acc_A** | 94.09% | ✅ 优秀（转向动作 94% 准确） |
| **Val Acc_M** | 93.15% | ✅ 优秀（鼠标 bucket 93% 准确） |
| **过拟合** | Train 99% vs Val 94% | ✅ 可控（差距 5%） |
| **训练时间** | 1.18 小时 | ✅ 速度达标 |

**3 个最佳模型已保存**：
```
checkpoints/goal_bc_v55_best_acc_a.pt  ← 推荐使用
checkpoints/goal_bc_v55_best_loss.pt
checkpoints/goal_bc_v55_best_acc_m.pt
```

---

## 模型架构

**文件**: `training/goal_conditioned_bc_v55_optimized.py`

**网络结构**:
- **Backbone**: ResNet18（修改 conv1 输入通道 3→12，支持 4 帧堆叠）
- **Goal Embedding**: 2 个目标（goal 0, goal 1）
- **Action Head**: 3 类（forward, turn_left, turn_right）
- **Mouse Head**: 7 类（bucket 0-6，映射到 mouse_dx）

**输入**: `[1, 12, 224, 224]`（4 帧×3 通道沿通道维度堆叠）

**输出**: 
- Action Logits: `[1, 3]`
- Mouse Logits: `[1, 7]`

---

## 推理脚本（已修复）

**文件**: `training/inference_goal_v55.py`

**修复内容**:
1. ✅ 模型加载参数（`num_goals=2, freeze_backbone=False`）
2. ✅ Checkpoint 格式（提取 `model_state_dict`）
3. ✅ Frame stacking 维度（`np.concatenate` → `[12, 224, 224]`）
4. ✅ 鼠标执行逻辑（高频微步 + 平滑滤波）

**启动命令**:
```powershell
cd D:\projects\wukong_ai
C:\Python\python.exe -u training\inference_goal_v55.py ^
  --model "D:\projects\wukong_ai\checkpoints\goal_bc_v55_best_acc_a.pt" ^
  --goal-id 1 ^
  --duration 60
```

---

## 已知问题

### 1. 鼠标输入可能无效（Raw Input 阻塞）⚠️

**问题**: 游戏使用 Raw Input API，过滤 Windows 级鼠标模拟（`pydirectinput.moveRel()` 无效）

**解决方案**（待验证）:
1. **ViGEmBus**（模拟 Xbox 手柄，右摇杆控制视角）
2. **Interception**（驱动级鼠标注入，需 Python 3.13+）

**当前状态**: 推理脚本已集成 `pydirectinput`，但游戏内可能无效。

### 2. 数据采集覆盖不足

**问题**: 
- 缺起始点转向样本
- 缺卡墙角恢复样本
- 缺精确转向量样本

**影响**: 推理时可能卡墙角或转向不精确。

---

## 项目文件结构

### 核心文件（保留）
```
wukong_ai/
├── training/
│   ├── goal_conditioned_bc_v55_optimized.py  # v5.5 训练脚本（推荐）
│   └── inference_goal_v55.py                # v5.5 推理脚本（已修复）
├── checkpoints/
│   ├── goal_bc_v55_best_acc_a.pt            # 最佳 Acc_A 模型（推荐）
│   ├── goal_bc_v55_best_loss.pt             # 最佳 Loss 模型
│   └── goal_bc_v55_best_acc_m.pt            # 最佳 Acc_M 模型
├── pathfinding_data_balanced/                # 平衡数据集（9,790 样本）
│   ├── pathfinding_ep1_*.h5
│   └── ...
├── docs/
│   ├── TECHNICAL_ANALYSIS.md                # 技术分析文档
│   └── RESEARCH_BC_FAILURE_ANALYSIS.md      # BC 失败分析
└── PROJECT_SUMMARY_v55.md                  # 本文件
```

### 冗余文件（可删除）
```
training/__pycache__/                        # 缓存文件
training_v55_optimized.log                   # 训练日志（28MB）
training_v55.log                             # 旧训练日志（2.3MB）
checkpoints/goal_bc_v55_epoch*.pt            # 周期检查点（已有最佳模型）
training/goal_conditioned_bc_v55.py         # 旧版训练脚本
training/goal_conditioned_bc_v55_fixed.py    # 旧版训练脚本
training/inference_goal_v55_standalone.py     # 独立版（已合并）
training/analyze_v55*.py                     # 分析脚本（临时）
training/oversample_dataset_v2.py            # 数据增强（已完成）
training/focal_loss.py                       # Focal Loss（已集成）
assist/                                      # 辅助脚本（临时）
WEEKEND_PLAN.md                              # 旧文档
TODO.md                                      # 旧文档
```

---

## 下一步计划

### 高优先级
1. ⏳ **测试推理效果**（游戏内验证）
2. ⏳ **解决鼠标输入问题**（ViGEmBus 或 Interception）
3. ⏳ **推送 v5.5 到 GitHub**

### 中优先级
4. ⏳ 采集更多转向数据（起始点、卡墙角恢复）
5. ⏳ DAgger 迭代（降低干预率至 <25%）
6. ⏳ 添加 LSTM 捕捉时序依赖

---

## 技术细节

### 训练配置
- **数据集**: `pathfinding_data_balanced/` (9,790 样本)
- **Batch Size**: 4（有效 batch=16，梯度累积=4）
- **优化器**: Adam (lr=1e-4, weight_decay=1e-4)
- **学习率调度**: CosineAnnealingLR (T_max=50, eta_min=1e-6)
- **损失函数**: Focal Loss（gamma=2.0，处理类别不平衡）
- **设备**: RTX 2060 6GB + 4.5GB RAM

### 数据增强
- **Oversample 非 bucket 3 样本**（bucket 3 占比从 81.15% → 46.27%）
- **总样本数**: 5,582 → 9,790 (1.75x)

---

## 参考文档

- `docs/TECHNICAL_ANALYSIS.md` - 技术分析文档
- `docs/RESEARCH_BC_FAILURE_ANALYSIS.md` - BC 失败分析
- `docs/VLA_Research.md` - VLA 架构调研
- `training/goal_conditioned_bc_v55_optimized.py` - 训练脚本（含详细注释）

---

**项目状态**: ✅ v5.5 训练完成，推理脚本已修复，待游戏内测试
**关键阻塞**: 鼠标输入问题（Raw Input）未解决，需 ViGEmBus 方案
**推荐模型**: `checkpoints/goal_bc_v55_best_acc_a.pt`
**推荐推理脚本**: `training/inference_goal_v55.py`
