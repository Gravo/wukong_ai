# wukong_ai 代码分析与改进方案

**分析日期**: 2026-05-19

---

## 一、当前方案问题诊断

### 1.1 核心问题：模型收敛但未学到有效动作

**现象**：
- 行为克隆准确率达到 74%，但推理时只会往前走
- 鼠标输出几乎恒定，视角控制失效
- 模型选择了"最安全"的策略：idle 或 forward

**根因分析**：

| 问题 | 原因 | 影响 |
|------|------|------|
| 数据分布失衡 | idle 33.4% + forward 54.2% = 87.6% | 模型学会"什么都不做"或"一直往前走" |
| Distribution Shift | BC 只学 state→action，不理解后果 | 偏离专家轨迹后错误累积 |
| 鼠标稀疏性 | 大部分帧鼠标不动，MSE 损失被淹没 | 模型学会输出接近 0 的值 |
| 损失函数不均衡 | 分类损失 vs 回归损失量级不同 | 鼠标部分学习不充分 |

### 1.2 内存泄漏问题（OOM）

**现象**：训练到 epoch 24 被 SIGKILL

**原因**：
1. 统计阶段加载完整 H5Dataset 只为计数（`behavior_clone_v2.py:96-109`）
2. CUDA 内存碎片化累积
3. Python GC 不及时释放引用

**已修复**：`gc.collect()` + `torch.cuda.empty_cache()` 已添加

### 1.3 代码结构问题

| 问题 | 文件 | 严重度 |
|------|------|--------|
| 模型定义重复 | `behavior_clone_v2.py` 和 `inference_v2.py` | 中 |
| 硬编码窗口坐标 | `config.py` | 低 |
| 无学习率调度 | `behavior_clone_v2.py` | 中 |
| 组合动作顺序执行 | `action_executor.py` | 高 |

---

## 二、替代方案研究

### 2.1 方案对比

| 方案 | 原理 | 适用性 | 复杂度 | 推荐度 |
|------|------|--------|--------|--------|
| **数据过滤 BC** | 只保留有动作的帧 | 简单有效 | 低 | ⭐⭐⭐⭐ |
| **DAgger** | 在线交互 + 专家纠正 | 需要人类在线 | 中 | ⭐⭐⭐⭐⭐ |
| **GAIL** | 对抗模仿学习 | 需要大量计算 | 高 | ⭐⭐⭐ |
| **Decision Transformer** | 序列建模，条件回报 | 需要回报标注 | 中 | ⭐⭐⭐ |
| **Diffusion Policy** | 扩散模型生成动作 | 高质量，慢推理 | 高 | ⭐⭐⭐⭐ |
| **OpenVLA** | 大规模预训练 VLA | 不适用游戏 | 很高 | ⭐ |

### 2.2 OpenVLA 分析（不推荐）

**架构**：
- 视觉编码器：DinoV2 + SigLIP
- 语言模型：Llama 2 7B
- 输入：图像 + 语言指令
- 输出：7 维机器人动作 token

**不适用原因**：
1. 动作空间不匹配：机器人 6DoF+夹爪 vs 游戏 WASD+鼠标
2. 数据需求过大：需 97 万轨迹预训练
3. 推理太慢：7B 模型无法实时
4. 语言指令不适用：游戏控制不需要语言条件

### 2.3 DAgger 分析（强烈推荐）

**原理**：
```
传统 BC: 专家数据 → 训练 → 部署（偏离专家轨迹 → 失败）
DAgger:  专家数据 → 训练 → 部署 → 收集新数据 → 专家标注 → 重新训练
```

**关键优势**：
- 训练数据覆盖模型实际会遇到的状态分布
- 解决 compounding error 问题
- 只需 2-3 轮迭代即可显著提升

**操作流程**：
1. 用当前 BC 模型控制角色
2. 人类玩家观察并实时按键纠正
3. 记录（模型状态, 人类纠正动作）对
4. 用这些数据重新训练

### 2.4 GAIL 分析（备选方案）

**原理**：
- Generator（策略网络）：状态 → 动作
- Discriminator（判别器）：(状态, 动作) → 是专家还是生成的？

**优点**：
- 不需要手动设计奖励
- 学到的策略更接近专家分布
- 可以处理多模态动作分布

**缺点**：
- 训练不稳定（GAN 的通病）
- 需要大量计算
- 对超参数敏感

### 2.5 Diffusion Policy 分析（理论最优）

**原理**：用扩散模型生成动作分布

**优点**：
- 能建模多模态动作分布（同一状态可有多种正确动作）
- 生成的动作更平滑
- 在机器人领域效果极好

**缺点**：
- 推理慢（需要多步去噪）
- 实现复杂

---

## 三、推荐实施路径

### 阶段 1：数据过滤 BC（立即可做，1-2 天）

**改动内容**：
1. 创建 `behavior_clone_v3.py`，实现 idle 帧过滤
2. 只保留 10-20% 的 idle 帧
3. 添加学习率调度（CosineAnnealingLR）
4. 加大鼠标损失权重（2.0x）
5. 增加训练轮数到 100 epochs

**预期效果**：
- 模型被迫学习有意义的动作
- 鼠标控制更敏感
- 训练更稳定

### 阶段 2：DAgger 在线学习（3-5 天）

**前提**：阶段 1 完成后，用改进的 BC 模型作为起点

**操作流程**：
1. 运行改进的 BC 模型控制角色（`inference_v2.py`）
2. 人类玩家观察并实时纠正（按 ESC 停止，按正确键）
3. 收集 10-20 分钟的纠正数据
4. 重新训练，重复 2-3 轮

**实现要点**：
- 需要修改 `data_collector.py` 支持 DAgger 模式
- 记录模型的动作和人类的纠正动作
- 训练时只用人类的纠正动作作为标签

### 阶段 3：如果 DAgger 不够，尝试 GAIL（1-2 周）

**需要实现**：
- 判别器网络
- PPO 训练循环
- 对抗训练稳定性调优

---

## 四、代码改动清单

### 4.1 新增文件

| 文件 | 说明 |
|------|------|
| `pathfinding/behavior_clone_v3.py` | 改进的 BC 训练脚本（数据过滤 + LR 调度） |
| `models/bc_model.py` | 共享的 BehaviorCloneModel 定义 |
| `docs/ANALYSIS.md` | 本文档 |

### 4.2 修改文件

| 文件 | 改动 |
|------|------|
| `config.py` | 添加 BC_V3 配置 |
| `pathfinding/inference_v2.py` | 使用共享模型定义 |
| `README.md` | 更新状态和下一步 |

### 4.3 已知问题修复

| 问题 | 状态 |
|------|------|
| 内存泄漏（OOM） | ✅ 已修复（gc.collect + cuda.empty_cache） |
| 模型定义重复 | ✅ 已提取到 models/bc_model.py |
| 无学习率调度 | ✅ 已添加 CosineAnnealingLR |
| 数据失衡 | ✅ 已实现 idle 帧过滤 |

---

## 五、在另一台机器上的操作步骤

### 5.1 环境准备

```bash
# 克隆仓库
git clone https://github.com/Gravo/wukong_ai.git
cd wukong_ai

# 安装依赖
pip install -r requirements.txt
```

### 5.2 数据准备

确保 `pathfinding_data/` 目录下有 h5 文件（从本机复制或重新录制）。

### 5.3 训练（改进版 BC）

```bash
# 使用改进的 BC v3（数据过滤 + LR 调度）
python pathfinding/behavior_clone_v3.py

# 或者使用原始 BC v2（如果想对比）
python pathfinding/behavior_clone_v2.py
```

### 5.4 推理测试

```bash
# 打开游戏，进入虎先锋战斗
python pathfinding/inference_v2.py --duration 120 --fps 10
```

### 5.5 DAgger 数据收集（阶段 2）

```bash
# 1. 运行推理，让模型控制角色
# 2. 人类观察并实时纠正
# 3. 收集纠正数据
python pathfinding/behavior_clone_v3.py --dagger-mode
```

---

## 六、参考资源

### 论文
- [DAgger](https://arxiv.org/abs/1011.0686) - Dataset Aggregation
- [GAIL](https://arxiv.org/abs/1606.03476) - Generative Adversarial Imitation Learning
- [Decision Transformer](https://arxiv.org/abs/2106.01345) - Reinforcement Learning via Sequence Modeling
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) - Visuomotor Policy Learning via Action Diffusion

### 代码库
- [OpenVLA](https://github.com/openvla/openvla) - Open Vision-Language-Action Model
- [imitation](https://github.com/HumanCompatibleAI/imitation) - Imitation Learning Library
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL Algorithms

---

## 七、总结

| 优先级 | 任务 | 预期收益 |
|--------|------|----------|
| P0 | 数据过滤 BC v3 | 立即提升动作多样性 |
| P1 | DAgger 在线学习 | 解决 distribution shift |
| P2 | GAIL / Diffusion Policy | 进一步提升（如果需要） |

**最实用路径**：数据过滤 BC → DAgger → 如果需要更强再考虑 GAIL

---

*文档生成时间: 2026-05-19*
