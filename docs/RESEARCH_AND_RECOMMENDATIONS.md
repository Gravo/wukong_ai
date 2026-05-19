# wukong_ai 深度研究与改进建议

**研究日期**: 2026-05-19  |  **深度等级**: 深度研究

---

## 一、核心问题诊断：为什么当前方案不行

### 1.1 问题本质

当前方案把「寻路」问题错误地定义为**纯视觉行为克隆**（pure visual BC）：

```
当前:  游戏画面 → 模型 → 动作
目标:  到达Boss房间
问题:  模型根本不知道「要去哪」，只能学「怎么动」
```

这导致两个根本性失败：

| 失败现象 | 本质原因 |
|---------|---------|
| 模型只会往前走 | idle+forward占了87.6%，这是最安全的不犯错策略，符合BC loss最小化 |
| 鼠标控制失效 | 鼠标是连续值，大多数帧dx=0，导致MSE被分类损失淹没 |
| 无法泛化到新场景 | BC只学到当前画面到动作的映射，偏离训练分布立即失效 |
| Compounding Error | 每一步小偏差累积，轨迹越来越偏离专家，最终完全错误 |

### 1.2 竞品对比：Turing-Project 的方案

GitHub上392星的 `Turing-Project/Black-Myth-Wukong-AI` 采用了完全不同的思路：

```
视觉模块：敌人RGB图像 → ResNet/CNN → 敌人位置/姿态
决策模块：DQN/PPO → 动作选择（基于血量+敌人状态）
跑图模块：LLM（GPT-4o/Claude）→ 自然语言探索指令
战斗核心：传统RL + 规则型状态机
```

**启示**：wukong_ai 应该借鉴他们的「解耦」思路，但用端到端视觉学习替代昂贵的LLM。

### 1.3 核心洞见

> 当前方案失败的根本原因是**缺少「目标」这个变量**。
> 模型不知道要去哪里，只能学到最安全的默认动作（idle/forward）。
> 只要加上goal标注，哪怕是最简单的one-hot，模型的表现就会有质的提升。

---

## 二、寻路问题的正确AI问题定义

### 2.1 问题重构：从「模仿动作」到「理解目标」

寻路的本质是：**给定当前画面 + 目标描述，到达目标位置**

四种定义层次：

```
Level 0（当前）:  state → action              【失败：不知道去哪】
Level 1          (state, goal_id) → action    【可行：目标条件BC】
Level 2          (state, goal_image) → action  【更好：目标图像对比】
Level 3          (state, semantic_goal) → action【最优：语义理解导航】
```

### 2.2 Level 1：目标条件行为克隆（Goal-Conditioned BC）

最简单的改进：给每帧标注「目标是什么」

采集时新增键位：
```
按 1 = 存档点
按 2 = Boss房间
按 3 = 岔路口
```
录制结束后用Boss血条出现/消失自动后标注。

模型架构：
```python
class GoalConditionedBC(nn.Module):
    def __init__(self, num_goals=5):
        self.encoder = ResNet18(pretrained=True)
        self.goal_emb = nn.Embedding(num_goals, 32)
        self.action_head = nn.Linear(256+32, NUM_ACTIONS)
        self.mouse_head = MouseHead()

    def forward(self, frame, goal_id):
        feat = self.encoder(frame)
        g = self.goal_emb(goal_id)
        combined = torch.cat([feat, g], dim=-1)
        return self.action_head(combined), self.mouse_head(feat)
```

### 2.3 Level 2：双流架构（最推荐）

借鉴Turing-Project的解耦思路但用端到端：

```
┌──────────────────────────────────────────────────────┐
│                  双流决策架构                            │
│                                                       │
│  流1 - 寻路（慢，2-5 FPS）：                             │
│    当前画面 + 目标画面 → 方向向量 [cos(theta), sin(theta)]│
│    训练：从存档点A到存档点B的无战斗移动轨迹                  │
│    效果：学会哪条路通向目标                               │
│                                                       │
│  流2 - 战斗（快，10+ FPS）：                             │
│    当前画面 + 敌人状态 → 动作序列                         │
│    训练：虎先锋战斗录制                                   │
│    效果：学会敌人出招 → 闪避/攻击                         │
│                                                       │
│  仲裁器：                                               │
│    if 检测到敌人 → 战斗流优先级                           │
│    else → 寻路流优先级                                   │
└──────────────────────────────────────────────────────┘
```

**为什么这个架构更好**：
- 寻路流只需要「朝哪个方向走」，不需要精确动作，泛化强
- 战斗流专门处理敌人出招反应，数据效率高
- 两流天然解耦，可以分别训练、分别改进

---

## 三、样本定义：从哪里来，到哪里去

### 3.1 当前样本的问题

| 问题 | 数据表现 | 影响 |
|------|---------|------|
| 无目标标注 | 每帧只有(action, frame) | 模型不知道要去哪 |
| 极度不平衡 | idle 33.4% + fwd 54.2% | 模型学会偷懒 |
| 无时序边界 | 视频流无叙事结构 | 无任务概念 |
| 鼠标稀疏 | 90%+帧dx=0 | 鼠标头无法有效学习 |

### 3.2 自动任务分段

利用游戏天然的任务边界来标注数据：

```
触发条件：
1. 土地庙检测（特定UI颜色/形状出现）→ 新任务开始
2. Boss血条消失（玩家击杀Boss）→ 任务成功
3. 玩家血量归零（死亡）→ 任务失败
4. 长时间无显著移动 → 可能卡住，segment结束

每个segment = 一个「任务」
```

---

## 四、可落地技术路线

### 路线A：快速改进（1-2天，立即可做）

在现有BC v3基础上打补丁：

**DAgger在线纠正**（最关键）：

```
Step 1: 用当前BC模型跑推理（inference_v2.py）→ 模型输出动作（每秒10次决策）
Step 2: 人类玩家在后台同时按键盘
         如果模型动作错误，人类按正确键
         记录：(state, human_correct_action)
Step 3: 人类纠正数据 → 重新训练BC模型
         重复2-3轮
```

DAgger解决的是BC的根本问题：**compounding error**。
BC训练时用的是「专家状态→专家动作」，但部署时模型会遇到「自己造成的非专家状态」。
DAgger把「模型遇到的状态」对应的「正确动作」补进训练集。

**鼠标控制修复**：

当前鼠标用MSE损失，被分类损失淹没。改进：
```python
# 鼠标单独一个头，单独加权
mouse_loss = SmoothL1Loss(mouse_pred, mouse_target) * 3.0  # 从2.0提高到3.0
# 鼠标方向归一化，大小用sigmoid非线性
direction = F.normalize(raw, dim=-1)
magnitude = torch.sigmoid(raw.abs().sum(dim=-1, keepdim=True) / 2)
```

### 路线B：目标条件BC（3-5天，值得做）

**采集改动**（data_collector.py）：
- 新增键位：1=存档点, 2=Boss房, 3=岔路口
- 按键时记录：当前帧的goal_id
- 录制结束后：自动用Boss血条出现/消失做后标注

**模型改动**（models/goal_conditioned_bc.py）：
- goal_embedding层（将goal_id映射到向量）
- forward增加goal参数
- 训练时：loss(state, goal, action)

### 路线C：双流架构（1-2周，有前景）

寻路流：
- 输入：当前画面 + 目标画面（或goal_id）
- 输出：方向向量 [cos(theta), sin(theta)]
- 训练数据：从存档点A到存档点B的无战斗移动
- 采集：在存档点按录制，走到下一个存档点停

战斗流：
- 输入：当前画面 + 敌人状态
- 输出：动作ID
- 训练数据：虎先锋战斗录制
- 复用现有数据和BC v3

### 路线D：扩散策略（2-4周，长期目标）

Diffusion Policy（ICRA 2024）相比BC的核心优势：**建模多模态动作分布**

```
BC:       P(action | state) = 单峰分布（比如只输出forward）
Diffusion: P(action | state) = 多峰分布（比如forward/left/dodge都可以）

对于战斗场景，同一画面可能有多种正确应对：
  敌人出拳 → 可以闪左、闪右、格挡、攻击打断
BC只能学到最常见的，Diffusion能保留所有合理选项
```

---

## 五、具体可执行任务清单

### P0：立即做（今天就能开始）

#### 1. DAgger数据采集（0.5天）

采集2-3轮纠正数据，每轮10-20分钟：

```bash
# 1. 用现有v2模型开始推理
python pathfinding/inference_v2.py --duration 300 --fps 10

# 2. 同时运行DAgger recorder（新增--dagger模式）
python training/data_collector.py --dagger --model checkpoints/bc_v2.pt --duration 600

# 3. 人类玩家：观察模型动作，如果错误，按正确键纠正
# 4. 训练：python pathfinding/behavior_clone_v3.py --dagger-data dagger_ep1.h5
```

预期收益：2-3轮后，compounding error问题显著改善。

#### 2. 修复鼠标控制（0.5天）

```python
class MouseHead(nn.Module):
    def forward(self, visual_feat):
        raw = self.mouse_net(visual_feat)
        # 鼠标方向归一化，大小保留非线性
        direction = F.normalize(raw, dim=-1)
        magnitude = torch.sigmoid(raw.abs().sum(dim=-1, keepdim=True) / 2)
        return direction * magnitude

# 损失函数：
cls_loss = CrossEntropyLoss(action_logits, action_labels) * 1.0
mouse_loss = SmoothL1Loss(mouse_pred, mouse_target) * 3.0
total = cls_loss + 3.0 * mouse_loss
```

### P1：本周做（2-5天）

#### 3. 目标条件BC（3天）

让模型知道要去哪：

采集改动：按1/2/3标注goal
模型改动：增加goal_embedding层
推理改动：启动时指定goal（--goal boss_door）

#### 4. 双流寻路-战斗分离（5天）

寻路流训练数据：从存档点到存档点的无战斗移动
战斗流训练数据：虎先锋战斗（复用现有）

### P2：下半月（2-4周）

#### 5. 拓扑图构建（2周）

自动发现游戏地图结构：
1. 用改进后的BC模型探索，记录每帧的视觉特征（ResNet/DINOv2）
2. 聚类：DBSCAN把相似画面聚成位置节点
3. 连接：相邻时间出现在一起的节点连边
4. 验证：用已知的存档点到Boss轨迹验证图的正确性

#### 6. 分层RL（4周）

高层：在拓扑图上A*搜索，产生子目标节点（每5秒决策一次）
低层：BC模型到达子目标，产生具体动作（每0.1秒决策一次）

---

## 六、参考资料

### 论文

| 论文 | 年份 | 关键思想 | 适用性 |
|------|------|---------|--------|
| DAgger | 2011 | 在线纠正数据聚合 | 立即可用 |
| GAIL | 2016 | 对抗模仿学习 | 可考虑 |
| Diffusion Policy | 2023 | 扩散模型生成动作 | 长期目标 |
| ACT | 2023 | Action Chunking Transformer | 战斗流可用 |
| MineDojo | 2023 | 开放世界AI agent | 架构参考 |
| Voyager | 2023 | LLM驱动Minecraft agent | 跑图思路 |
| RT-2 | 2023 | VLA端到端控制 | 长期愿景 |

### GitHub参考

| 仓库 | Stars | 关键内容 | 参考价值 |
|------|-------|---------|---------|
| Turing-Project/Black-Myth-Wukong-AI | 392 | DQN/PPO+ResNet+LLM跑图 | 架构思路 |
| anonymous4213/game-video-operation-align | 0 | VLM理解黑神话玩家动作 | VLM应用 |
| ikostrikov/pytorch-a2c-ppo-acktr-gail | 3901 | PPO/GAIL全家桶 | GAIL参考 |
| hcnoh/gail-pytorch | 177 | GAIL简洁实现 | GAIL入门 |
| zsdonghao/Imitation-Learning-Dagger-Torcs | 71 | DAgger实现 | DAgger参考 |
| anassee15/vision-voyager | 1 | VLM驱动Minecraft agent | 端到端VLM |

---

## 七、推荐实施路径

```
Week 1:  DAgger数据采集（3轮）+ BC v3训练 + 鼠标头修复
         |
Week 2:  目标条件BC上线（goal标注采集 + 模型改动）
         |
Week 3:  双流架构上线（寻路流 + 战斗流分离训练）
         |
Week 4:  拓扑图构建（自动发现位置节点）
         |
Week 5+: 分层RL（如果前面效果不够好）
         |
长期:    Diffusion Policy 或 轻量VLA
```

**最核心的一句话**：

> 模型不知道要去哪里，就永远只会做最安全的事：什么都不做或一直往前走。给模型一个目标，哪怕只是一个one-hot编码，表现就会有质的提升。

---

*研究完成时间: 2026-05-19*
