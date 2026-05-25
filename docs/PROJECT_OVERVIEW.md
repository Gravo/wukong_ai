# wukong_ai 项目全景：让高人10分钟看懂，给你最精准的建议

> **项目**: 用AI打《黑神话：悟空》（端到端视觉导航）  
> **GitHub**: https://github.com/Gravo/wukong_ai  
> **硬件**: RTX 2060 6GB显存 / 4.5GB可用RAM / Python 3.10 / PyTorch 2.3.1+cu121  
> **交互**: dxcam截图 + pydirectinput键鼠操作，非API接入  
> **作者有RL经验**: 连连看/贪食蛇/CartPole/飞翔的小鸟

---

## 一、我想做什么

让AI看游戏截图，自动操控键盘鼠标，从A点走到B点（寻路）。

**不**涉及：战斗、Boss、复杂策略。先解决最基础的「走到目标点」。

**关键约束**：游戏没有API，只能通过截图和键鼠交互（dxcam + pydirectinput）。

---

## 二、技术路线演进（走过的路）

### Phase 1: DQN打只狼（已废弃）
- 从旧项目迁移，代码有致命bug，全部重写

### Phase 2: PPO + ResNet18（方案C全量重写）
- 20个Python文件，已推GitHub
- 计划4阶段：数据采集 → 预处理 → 行为克隆训练 → 推理集成
- **卡在阶段2-3之间**：8249帧数据，预处理反复OOM/SIGKILL（4.5GB RAM不足）

### Phase 3: Goal-Conditioned Behavior Cloning
- 发现BC失败根因：**缺少goal变量**，模型不知道要去哪
- 添加goal embedding，让模型知道"当前目标是Goal 1还是Goal 2"
- 重新采集带goal标注的数据（data_collector_v3.py）

### Phase 4: 训练优化（v3 → v4 → v4.1 → v5 → v5.1）
- 反复迭代，始终未能解决核心问题

---

## 三、所有实验结果（数据说话）

### v3.0 / v3.1: Goal-Conditioned BC（连续鼠标回归）

| 版本 | 数据量 | 结果 | 问题 |
|------|--------|------|------|
| v3.0 | ~5000帧 | OOM killed | 显存不足 |
| v3.1 | ~5000帧 | Acc 99%+, 推理差 | 鼠标归一化/224导致输出±0.03，修复后±1.05 |

**v3.1推理表现**: 有明显转向但效果不行，Goal Embedding已生效（Goal1不怎么转，Goal2会转），但存在卡墙角死循环、转向抖动、起始点无转向。

### v4.1: 连续鼠标回归（修复版）

| 指标 | 值 |
|------|-----|
| 有效样本 | 4072 |
| 转向动作帧 | 755 (18.5%) |
| Acc | 99.66% |
| 推理效果 | 差 |

**问题**: Acc 99%但推理失败——典型协变量漂移。

### v5: 5类离散动作分类

```
动作映射: 0→idle, 4→forward, 5→turn_right, 6→turn_left, 3→dodge
架构: ResNet18 + goal embedding → 5-class CrossEntropyLoss + class weights
数据: 9212样本 (idle 39.0%, forward 52.7%, turn_left 3.2%, turn_right 5.0%, dodge 0%)
```

| Epoch | Loss | Acc |
|-------|------|-----|
| 1 | - | ~50% |
| 3 | - | ~52% |
| 10 | 5.31 | 63.44% |

**问题**: Acc卡在~63%，且主要是"总是预测idle/forward"。**模型试图学习转向但失败**。

### v5.1: 4帧堆叠（12通道输入）

```
修改ResNet18第一层conv: nn.Conv2d(12, 64, ...) (前3通道复制预训练权重)
batch_size: 4 (从8降低以适应显存)
```

| Epoch | Loss | Acc | Time |
|-------|------|-----|------|
| 1 | 1.3897 | 52.69% | 289.5s |
| 2 | 1.3348 | 52.78% | 303.4s |
| 3 | 1.3109 | 52.71% | 304.1s |
| 4 | 1.3166 | 52.74% | 304.9s |

**结论**: **帧堆叠完全没有帮助**。Loss下降但Acc完全停滞在52.7%。

### DAgger实验

| 轮次 | 数据 | 干预率 | 结果 |
|------|------|--------|------|
| Round 1-2 | 已删除 | - | 数据无效 |
| Round 3 | 199帧干预 | 71.5% | Acc 79.45% |
| v4.1修复后 | 4072+1135=5820样本 | - | Acc 63.44% |

**DAgger发现的问题**:
- 83%干预帧是idle（数据采集时模型大部分时间在"不动"）
- action编码错误：原始4/5/6被误当作ASD类别
- 鼠标数据未记录（ctypes缺失导致全0）

---

## 四、数据现状

### 数据分布（核心矛盾）

```
idle:    3573 (39.0%)
forward: 4827 (52.7%)
turn_L:   290 (3.2%)
turn_R:   462 (5.0%)
dodge:      0 (0.0%)
```

**问题**: 92%是idle+forward。模型"全猜forward"就有52.7%准确率，学转向的收益太低。

### 数据来源
- 20个h5文件（pathfinding_data/目录）
- pathfinding_dagger_round3.h5（DAgger纠正数据）
- 总可用样本: ~9152帧（人类5584 + DAgger 1135 + 其他补充）

---

## 五、失败根因（我的判断 + 证据）

### 判断1: 视觉信息不足
**论点**: "是否转向"的决策无法从截图中提取。

**证据**:
- v5.1帧堆叠（4帧=1.3秒历史）Acc仍52.7%
- 人类玩家依据：小地图、目标指示器、3D空间感知、肌肉记忆
- 同一截图在不同时刻可能对应不同动作（取决于玩家意图）

**反驳空间**: 帧堆叠可能还不够长？4帧只有1.3秒，也许需要30帧（10秒）？但显存不允许。

### 判断2: 协变量漂移（已确认）
**论点**: BC学的是 P(a|s) where s~专家分布，推理时s~模型分布。

**证据**:
- v4.1 Acc 99%+ 推理差（教科书式协变量漂移）
- DAgger理论可解，但需要多轮迭代（当前只做了1轮，干预率71.5%→需降到<25%）

### 判断3: 数据不均衡是结果而非原因
**论点**: 数据不均衡不是BC失败的原因，而是"人类演示大部分时间在idle/forward"的客观反映。

**证据**: 即使加了class weights（非idle类别5x），Acc仍卡住。说明模型**确实学不到转向信号**。

### 判断4: 标签噪声（已修复但影响有限）
**论点**: DAgger数据中action编码错误（4/5/6被误当作ASD），鼠标数据全0。

**证据**: 已在v4.1修复，但修复后Acc仍63%。说明这不是主因。

---

## 六、研究过的相关方案

### 已调研但未实现

| 方案 | 论文/来源 | 评估 |
|------|----------|------|
| PIDM (Predictive Inverse Dynamics) | arXiv:2601.21718 (2026) | 论文证明在3D游戏环境BC需多66%样本 |
| World Models | arXiv:1803.10122 (Ha & Schmidhuber) | world_model.py已写但未与BC结合 |
| VLA (Vision-Language-Action) | OpenDriveVLA/OpenVLA | 算力不够（7B参数） |
| Decision Transformer | Chen et al. 2021 | 需要设计reward，且需大量数据 |
| LSTM时序建模 | 经典方法 | RTX 2060显存可能不够 |
| 自动子目标发现 | 聚类方法 | TODO.md有计划但未实施 |
| 辅助任务（深度预测等） | GridToPix 2021 | 需要额外的ground truth |
| Active DAgger (RND) | Bire et al. 2025 | 改进DAgger效率 |
| MoE分层架构 | 自研路线 | 战斗专家+寻路专家 |
| 自适应帧跳过 | "老花眼开车"洞察 | 优化效率不改变能力 |

### 已实现

| 组件 | 文件 | 状态 |
|------|------|------|
| 数据采集器v3 | data_collector_v3.py | 可用，支持goal_id |
| DAgger采集器 | dagger_collector.py | F10干预/ESC退出 |
| 训练v5/v5.1 | goal_conditioned_bc_v5.py / v51.py | 可用但效果差 |
| 懒加载训练 | digger_training.py v2.0 | 解决OOM |
| 推理v4/v5 | inference_goal_v4.py / v5.py | 可用 |
| L2辅助驾驶 | AutoDodge/AutoFace/Arbitrator | 可用（commit 44ec4f9） |
| PointNav模型 | DD-PPO+MiDaS深度 | 可用（commit ca42714） |
| 世界模型 | world_model.py | 已实现但未集成 |
| goal_id转换 | convert_goal_id_to_zero_indexed.py | 已批量转换20个h5文件 |

---

## 七、当前代码架构

```
D:\projects\wukong_ai\
├── data_collector_v3.py          # 数据采集（dxcam截图 + keyboard库）
├── dagger_collector.py           # DAgger干预数据采集（F10/ESC）
├── training/
│   ├── goal_conditioned_bc.py    # v3 基线
│   ├── goal_conditioned_bc_v4.py # v4.1 鼠标修复
│   ├── goal_conditioned_bc_v5.py # v5 离散5类
│   ├── goal_conditioned_bc_v51.py # v5.1 帧堆叠
│   ├── digger_training.py       # DAgger训练（懒加载）
│   ├── inference_goal.py        # 推理脚本
│   ├── inference_goal_v4.py     # v4推理
│   └── inference_goal_v5.py     # v5推理
├── pathfinding_data/            # 20个h5文件 (~9152帧)
├── pathfinding_dagger_round3.h5 # DAgger纠正数据
├── checkpoints/                  # 模型checkpoint
├── docs/
│   ├── TECHNICAL_ANALYSIS.md    # 技术分析（916行）
│   ├── VLA_Research.md          # VLA调研
│   ├── RESEARCH_FOUNDATION_MODELS.md # 基础模型研究
│   └── RESEARCH_BC_FAILURE_ANALYSIS.md # BC失败分析
└── TODO.md                       # 路线图（部分过时）
```

---

## 八、硬件约束（很重要）

| 资源 | 规格 | 影响 |
|------|------|------|
| GPU | RTX 2060 6GB | 不能训练大模型（LSTM、7B VLA） |
| RAM | 4.5GB可用 | 不能全量加载h5到内存 |
| CPU | 未知 | 帧堆叠每epoch需5分钟（2288 batches） |
| 训练速度 | ~300s/epoch (v5.1) | 30 epochs需要2.5小时 |
| 截图工具 | dxcam | 仅限NVIDIA GPU，速度快 |

---

## 九、我想听到的建议（具体到可执行）

请针对以下问题给建议：

### Q1: BC路线还有救吗？
- 我尝试了：连续回归、离散分类、帧堆叠、DAgger、class weights、EMA平滑
- 结果全部失败（Acc卡在52-63%）
- **有没有我没试过的BC技巧可以救？**

### Q2: PIDM是正确方向吗？
- 论文arXiv:2601.21718说PIDM在3D游戏环境中比BC高66%样本效率
- 但PIDM需要预测未来状态，在黑神话这种复杂3D场景中，未来状态预测本身也很难
- **PIDM真的适合这个场景吗？**

### Q3: 应该彻底换方法吗？
- 如果BC路线确实走不通，应该换什么？
- RL？PPO？还是别的什么？
- 在RTX 2060 + 4.5GB RAM的约束下，什么方案是可行的？

---

## 十、同类项目参考

| 项目 | Stars | 方法 | 备注 |
|------|-------|------|------|
| Turing-Project/Black-Myth-Wukong-AI | 392 | DQN/PPO+ResNet战斗 | 只做战斗不做寻路 |
| OpenAI Five | - | PPO (5v5 Dota2) | 大规模分布式训练 |
| AlphaStar | - | 行为克隆+RL (StarCraft2) | Google级别算力 |
| MineRL | - | 模仿学习 (Minecraft) | 有专门的数据集和竞赛 |
| AgentSandbox | - | 多游戏通用框架 | 学术研究用 |

**关键观察**: 没有找到"端到端视觉寻路打3D动作游戏"的成功开源项目。最近的参考是MineRL（Minecraft寻路），但Minecraft的视觉复杂度远低于黑神话。

---

*最后更新: 2026-05-25*  
*项目周期: 2026-05-14 至今（12天）*
