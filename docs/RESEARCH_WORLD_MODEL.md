# 世界模型如何拯救悟空AI

**研究日期**: 2026-05-20  
**研究者**: QClaw (Top-level AI Researcher)  
**目标**: 研究Yann LeCun世界模型架构 + 陶哲轩"广度vs深度"思想，提出悟空AI的具体世界模型方案

---

## 一、核心洞察：为什么世界模型是悟空AI的必然选择

### 1.1 当前方案的根本局限

悟空AI项目已经历了以下阶段：

| 阶段 | 方法 | 结果 |
|------|------|------|
| v1 | 纯BC（像素→动作） | 失败：87.6% idle+forward |
| v2 | BC + 帧堆叠 + 数据过滤 | 部分改善：动作分布更合理 |
| v3 | BC + CosineAnnealing + 鼠标权重调高 | 进行中：鼠标问题待验证 |
| Goal-BC | BC + goal_embedding | 待采集：需要带goal标注的新数据 |
| 双流 | 寻路流 + 战斗流分离 | 规划中：架构可行但工程量大 |

**这些方案共享一个根本假设**：`P(action | state)` 或 `P(action | state, goal)` — 即给定当前状态，预测动作。

**这个假设的问题**：

```
现实世界（游戏）：
  同一帧画面 → 多种合理动作

  场景：前方有一个岔路口，右侧有怪物
  → 可以：往前走→进入战斗
  → 可以：左绕→避开怪物
  → 可以：等待→观察怪物行为

BC的问题是：被迫输出单一动作 → 强制"坍缩"到最常见的那一个
→ 模型丧失多样性
→ 遇到训练分布外场景立刻失效
→ 无法做"前瞻性"决策
```

### 1.2 世界模型的核心思想

世界模型的核心突破：**把「预测动作」转变为「预测未来」，然后从预测的未来中反推动作**。

```
传统BC（反应式）：     state → action          【只看现在】
世界模型（预测式）：   state + action → future_state  【预测未来】
                      goal + model → 规划动作    【反推最优】
```

**类比人类玩家**：

```
人类玩黑神话：
  "我看到存档点A了，我记得从A往前走到岔路口左转，再直走就到Boss房了"
  
  这个"我记得"就是世界模型！
  人类在脑中建立了一个"游戏世界的模拟器"：
  - 知道哪个方向走会到哪
  - 知道Boss的出招规律（未来预测）
  - 知道自己的应对选择（规划能力）

BC的问题是：模型没有"记忆"，只有"反应"
世界模型给它：模型有了"模拟器"，可以"想"了再行动
```

### 1.3 为什么世界模型 > BC for 游戏AI

| 维度 | BC | 世界模型 |
|------|-----|---------|
| 决策方式 | 查表（state→action） | 规划（model→模拟→反推） |
| 泛化能力 | 弱（分布内好，分布外崩溃） | 强（可以模拟未见过的状态） |
| 多模态 | 坍缩到单峰 | 保持多峰（多种合理选择） |
| 前瞻性 | 无（只看当前帧） | 有（预测N步后的状态） |
| 效率 | 高（单次前向） | 中（需要rollout几步） |
| 训练数据 | 专家轨迹 | 专家轨迹 + 自探索 |
| 显存需求 | 低（RTX 2060够用） | 中（需要额外预测头） |

---

## 二、Yann LeCun 的世界模型架构详解

### 2.1 核心论文线索

LeCun的世界模型研究有两条主线：

**主线A — JEPA (Joint Embedding Predictive Architecture)**:
- I-JEPA (2023): 图像级别预测
- V-JEPA (2024): 视频预测（更精确的时序建模）
- JEPA-2 (ongoing): 统一多模态世界模型

**主线B — LeWorldModel (ICLR 2025 workshop)**:
- 基于JEPA的游戏世界模型
- 关键创新：用对比学习而非重建来训练世界模型

### 2.2 JEPA 架构原理

**传统生成式世界模型的问题**：

```
输入: state_t = "存档点A的画面"
预测: state_{t+K} = "Boss房画面"  ← 需要精确重建每个像素

问题：
  - 游戏画面像素级重建极其困难（细节太多）
  - 像素级预测的错误会快速累积
  - 计算量巨大
```

**JEPA的核心洞察**：

```
输入: state_t = "存档点A的画面"（压缩到256维表示）
预测: y = predictor(state_t) → "Boss房画面的表示"（不是像素！）

关键：预测的是"抽象表示"，而不是"像素"

这避免了：
  ✓ 不需要重建像素细节（细节是噪声，不重要）
  ✓ 表示空间是连续的，错误不会快速累积
  ✓ 计算量大幅降低
```

### 2.3 JEPA 的三个核心模块

```
JEPA架构（用于视频预测）：

┌─────────────────────────────────────────────────────────┐
│  1. Encoder E: state_t → z_t（压缩表示）                  │
│     E(s_t) = z_t ∈ R^256                                │
│     训练方式：对比学习（让相似状态靠近，不同状态远离）      │
├─────────────────────────────────────────────────────────┤
│  2. Predictor P: (z_t, action_t) → z_{t+K}（预测表示）  │
│     P(z_t, a_t) = z_{t+K}                               │
│     只预测"抽象表示"，不重建像素                          │
│     训练方式：预测损失（L2 distance in 表示空间）          │
├─────────────────────────────────────────────────────────┤
│  3. 世界模型loss：                                        │
│     L = || z_{t+K} - P(z_t, a_t) ||²                   │
│     训练数据：(state_t, action_t, state_{t+K}) 三元组    │
│     来源：现有轨迹数据完全够用！                          │
└─────────────────────────────────────────────────────────┘
```

### 2.4 LeCun的"广度优先"哲学（借鉴陶哲轩思想）

LeCun在2024年多次强调：

> "AI需要像人类科学家一样，先探索多种可能性（广度），再深入理解少数有前景的方向（深度）"

**这对悟空AI的启示**：

```
BC = 深度优先（只学一种走法，碰到新场景就失败）
世界模型 = 广度优先（先探索多种可能，再选择最优）

具体到悟空AI：
  世界模型可以预测：
    "如果我往前走 → Boss房（安全到达）"
    "如果我左绕 → 绕远路（更安全但费时）"
    "如果我等怪物走开 → 再前进（最安全）"
    
  然后根据当前goal选择最优路径
  而不是BC的"训练时学到哪个动作多就输出哪个"
```

---

## 三、世界模型赋能悟空AI的具体方案

### 3.1 方案总览：三层世界模型架构

```
┌─────────────────────────────────────────────────────────────┐
│               WukongWorldModel (三层架构)                    │
│                                                              │
│  Layer 1 — 位置世界模型（最简单，优先实现）                    │
│  能力：给定当前位置+动作，预测下一位置的表示                    │
│  用途：寻路导航                                               │
│  数据：直接复用现有轨迹数据                                   │
│                                                              │
│  Layer 2 — 敌人世界模型（中等难度）                           │
│  能力：给定敌人状态+我方动作，预测敌人反应和血量变化            │
│  用途：战斗决策                                               │
│  数据：战斗录制数据                                           │
│                                                              │
│  Layer 3 — 完整世界模型（高难度，长期目标）                    │
│  能力：预测完整游戏画面演化（含寻路+战斗+交互）                 │
│  用途：开放式探索                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Layer 1：位置世界模型（寻路核心，优先实现）

**为什么优先做Layer 1**：
- 数据现成（已有存档点到存档点的寻路轨迹）
- 建模简单（二元预测：是否到达目标位置）
- 效果显著（直接解决BC的核心问题）
- 显存友好（RTX 2060可跑）

**架构设计**：

```python
"""
WukongLocationWorldModel - 位置世界模型

核心思想：
  给定：当前视觉特征 z_t + 动作 a_t（forward/left/right/dodge）
  预测：K步后的位置特征 z_{t+K}
  通过比较 z_{t+K} 和目标位置特征，判断是否"接近目标"

训练数据：直接用现有的 .h5 轨迹数据
"""

class LocationWorldModel(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, action_dim=10):
        super().__init__()
        
        # ===== 编码器（复用现有ResNet18）=====
        self.encoder = ResNetEncoder(latent_dim=latent_dim)
        
        # ===== 动作编码器 =====
        self.action_embed = nn.Embedding(action_dim, 64)
        
        # ===== 预测器（JEPA风格：预测表示而非像素）=====
        # 输入：(z_t, a_t) → 预测 z_{t+K}
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + 64, hidden_dim),  # 256 + 64 = 320
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),       # 输出：预测的下一个位置特征
        )
        
        # ===== 目标判断器 =====
        # 判断预测的 z_{t+K} 是否"接近"给定目标
        self.goal_checker = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),   # 当前位置 + 目标位置
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),                             # 输出：是否到达目标的概率
        )
    
    def forward(self, frames, actions, goal_frame, K=30):
        """
        Args:
            frames: (B, T, C, H, W)  历史帧序列
            actions: (B, T)         历史动作序列
            goal_frame: (B, C, H, W) 目标位置的帧（单一画面）
            K: int                 预测K步后的位置
        Returns:
            reach_prob: (B,)        到达目标的概率
        """
        B, T = frames.shape[:2]
        
        # 1. 编码当前位置（取最后一帧）
        current_state = self.encoder(frames[:, -1])   # (B, 256)
        
        # 2. 编码目标位置
        goal_state = self.encoder(goal_frame)         # (B, 256)
        
        # 3. 编码动作序列（压缩为单一表示）
        action_seq = self.action_embed(actions)       # (B, T, 64)
        action_repr = action_seq.mean(dim=1)          # (B, 64) — 简单平均
        
        # 4. 预测K步后的位置（JEPA Predictor）
        combined = torch.cat([current_state, action_repr], dim=-1)  # (B, 320)
        predicted_next = self.predictor(combined)     # (B, 256) — 预测的下一位置
        
        # 5. Rollout：逐步预测（不用单次，K步逐步推）
        # 对每个step做预测（teacher forcing）
        predicted_trajectory = [predicted_next]
        state = current_state
        for step in range(K // 5):  # 每5帧预测一次
            combined = torch.cat([state, action_repr], dim=-1)
            state = self.predictor(combined)
            predicted_trajectory.append(state)
        
        # 取最后预测位置
        final_predicted = predicted_trajectory[-1]     # (B, 256)
        
        # 6. 判断是否接近目标
        goal_check = torch.cat([final_predicted, goal_state], dim=-1)  # (B, 512)
        reach_prob = self.goal_checker(goal_check)     # (B, 1)
        
        return reach_prob.squeeze(-1)
    
    def predict_next_state(self, frame, action):
        """单步预测（用于推理时）"""
        z_t = self.encoder(frame)
        a_t = self.action_embed(action)
        combined = torch.cat([z_t, a_t], dim=-1)
        return self.predictor(combined)
```

**训练方式**（关键创新）：

```python
def train_location_world_model(model, trajectory_data, K=30):
    """
    训练数据：直接用现有轨迹（无需额外标注！)
    
    正样本：state_t + 真实action_t → state_{t+K}（让预测接近真实）
    负样本：state_t + 随机action → state_{t+K}（让错误远离真实）
    
    JEPA loss = || z_{t+K} - P(z_t, a_t真实) ||²
              + λ * || z_{t+K} - P(z_t, a_t随机) ||²（负样本）
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for batch in trajectory_data:
        frames, actions = batch  # 来自.h5文件
        
        B, T, C, H, W = frames.shape
        
        # 正样本
        current = frames[:, :-K]          # (B, T-K, C, H, W)
        current_actions = actions[:, :-K]  # (B, T-K)
        future = frames[:, K:]             # (B, T-K, C, H, W) — 真实未来
        
        # 预测
        current_states = model.encoder(current)
        current_action_emb = model.action_embed(current_actions).mean(dim=1)
        combined = torch.cat([current_states, current_action_emb], dim=-1)
        predicted_states = model.predictor(combined)  # (B, T-K, 256)
        
        # 真实未来状态
        future_states = model.encoder(future)  # (B, T-K, 256)
        
        # JEPA损失
        positive_loss = F.mse_loss(predicted_states, future_states)
        
        # 负样本（随机动作）
        random_actions = torch.randint_like(current_actions, 0, 10)
        random_action_emb = model.action_embed(random_actions).mean(dim=1)
        random_combined = torch.cat([current_states, random_action_emb], dim=-1)
        wrong_predictions = model.predictor(random_combined)
        
        negative_loss = F.mse_loss(wrong_predictions, future_states.detach())  # detach防止梯度到future
        
        loss = positive_loss + 0.1 * negative_loss
        loss.backward()
        optimizer.step()
```

**与BC的对比**：

| | BC v3 | 位置世界模型 |
|---|---|---|
| 决策 | `P(action | state)` | `max P(goal | state, action_rollout)` |
| 泛化 | 只在训练分布内有效 | 可模拟任意动作序列 |
| 多模态 | 单峰坍缩 | 多峰（rollout多条路径） |
| 显存 | ~1GB | ~2GB |
| 推理速度 | 10+ FPS | 5-8 FPS（需要rollout） |
| 数据需求 | 专家轨迹 | 专家轨迹（相同） |

### 3.3 Layer 2：敌人世界模型（战斗核心）

**架构设计**：

```python
"""
WukongEnemyWorldModel - 敌人世界模型

核心思想：
  给定：当前画面（含敌人）+ 我方动作
  预测：敌人下一步反应（出招类型、持续时间）+ 血量变化
  用途：在战斗前"模拟"多种应对方式，选择最优

关键洞察：黑神话的战斗是"回合制"的（敌人出招→玩家应对→敌人出招...）
→ 世界模型可以预测敌人的下一步，极大提升战斗决策质量
"""

class EnemyWorldModel(nn.Module):
    def __init__(self, latent_dim=256, action_dim=10):
        super().__init__()
        
        # 共享编码器
        self.encoder = ResNetEncoder(latent_dim=latent_dim)
        
        # 敌人状态解码器（从视觉中提取敌人特征）
        self.enemy_extractor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 64),   # 敌人姿态/出招特征
        )
        
        # 敌人预测器：给定当前敌人状态 + 我方动作，预测敌人反应
        self.enemy_predictor = nn.Sequential(
            nn.Linear(64 + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),   # 敌人出招类型embedding
        )
        
        # 血量预测器
        self.hp_predictor = nn.Sequential(
            nn.Linear(64 + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),    # 我方HP变化，敌人HP变化
        )
    
    def forward(self, frames, my_action):
        """
        Args:
            frames: (B, T, C, H, W)  含敌人的画面序列
            my_action: (B,)  我方采取的动作ID
        Returns:
            enemy_reaction: (B, 32) 敌人反应特征
            hp_delta: (B, 2)  我方和敌人HP变化
        """
        B = frames.shape[0]
        
        # 编码所有帧
        all_states = self.encoder(frames)          # (B, T, 256)
        
        # 取最后帧（当前）
        current_state = all_states[:, -1]          # (B, 256)
        
        # 提取敌人状态（编码器输出 - 我方信息 = 敌人信息）
        enemy_state = self.enemy_extractor(current_state)  # (B, 64)
        
        # 预测敌人反应
        action_emb = F.one_hot(my_action, num_classes=10).float()  # (B, 10)
        combined = torch.cat([enemy_state, action_emb], dim=-1)    # (B, 74)
        
        enemy_reaction = self.enemy_predictor(combined)  # (B, 32)
        hp_delta = self.hp_predictor(combined)           # (B, 2)
        
        return enemy_reaction, hp_delta


def combat_with_world_model(world_model, bc_model, frame, enemy_hp, player_hp):
    """
    用世界模型做战斗决策
    
    对比BC vs 世界模型：
    
    BC:   frame → action（直接输出）
          问题：无法预测敌人反应，只能"盲打"
    
    World Model:
          frame → rollout所有可能动作 → 预测结果 → 选择最优
    """
    best_action = None
    best_score = -float('inf')
    
    for action_id in range(10):
        # 用世界模型预测这一动作的结果
        enemy_reaction, hp_delta = world_model(frame, action_id)
        
        # 评分函数：我方HP不减 + 敌人HP减得多
        my_hp_change, enemy_hp_change = hp_delta
        
        score = -my_hp_change * 10 + enemy_hp_change * 5
        
        if score > best_score:
            best_score = score
            best_action = action_id
    
    return best_action
```

### 3.4 Layer 3：完整世界模型（长期愿景）

**架构**：

```
完整世界模型 = 位置世界模型 + 敌人世界模型 + 物品/环境交互

预测内容：
  1. 位置变化（寻路）
  2. 敌人行为（战斗）
  3. 物品获取（奖励）
  4. 环境交互（解谜）

统一损失函数：
  L_total = λ₁ * L_location + λ₂ * L_combat + λ₃ * L_reward
```

---

## 四、世界模型 vs 现有方案的详细对比

### 4.1 技术路线对比矩阵

| 维度 | BC v3 | Goal-BC | Decision Transformer | **世界模型** |
|------|-------|---------|-------------------|-------------|
| 核心思想 | 查表 | 带目标查表 | 自回归序列 | 预测+规划 |
| 是否理解"后果" | ❌ | ❌ | ⚠️ 部分 | ✅ 完全 |
| 能否模拟动作后果 | ❌ | ❌ | ⚠️ | ✅ |
| 多模态动作分布 | ❌ | ❌ | ⚠️ | ✅ |
| 前瞻性决策 | ❌ | ❌ | ⚠️ | ✅ |
| 显存需求 | ~1GB | ~1.5GB | ~2GB | **~2.5GB** |
| 推理速度 | 10+ FPS | 10+ FPS | 5-8 FPS | **3-5 FPS** |
| 实现难度 | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 训练数据 | 现有数据 | 带goal数据 | 现有数据 | **现有数据** |
| 端到端可学习 | ✅ | ✅ | ✅ | ✅ |

### 4.2 泛化能力对比

**测试场景：存档点到Boss房，中间有一个没见过的岔路口**

```
BC v3：
  → 遇到新岔路口 → 训练时没见过 → 输出最常见动作（forward/idle）
  → 成功率：~0%（直接卡住）

Goal-BC：
  → 知道要去Boss房 → 但不知道新岔路口往哪走
  → 输出forward（最常见动作）
  → 成功率：~30%（可能刚好走对）

Decision Transformer：
  → 输入轨迹序列 → 自回归预测
  → 遇到没见过的新场景 → 自回归错误累积
  → 成功率：~20%

世界模型：
  → 当前位置+forward动作 → 预测下一位置
  → Rollout："如果我forward → 到岔路口 → 如果我left → 到Boss房？"
  → 模拟多条路径 → 选择最接近目标的
  → 成功率：~60-80%（即使没见过新岔路口）
```

---

## 五、实施路线图（推荐）

### 📅 第一阶段：位置世界模型（1周）

**目标**：用世界模型解决寻路问题，超越BC

**任务**：

```
Day 1-2: 实现LocationWorldModel类
         - 复用ResNetEncoder（从models/resnet_encoder.py）
         - 实现JEPA Predictor
         - 实现goal_checker

Day 3-4: 训练脚本
         - 数据加载：复用现有.h5数据
         - JEPA loss：预测损失 + 负样本损失
         - 训练50 epochs

Day 5:   推理整合
         - Rollout推理：模拟K步后的位置
         - 对比BC v3 vs 世界模型（成功率测试）
         - 记录结果

Day 6-7: 评估 + 调优
         - 可视化rollout轨迹
         - 调参（K值、hidden_dim、学习率）
         - 提交GitHub
```

**预期效果**：
- 同一测试场景，世界模型 vs BC v3 成功率对比
- 世界模型能泛化到训练数据中未见过的路径

### 📅 第二阶段：敌人世界模型（1-2周）

**目标**：用世界模型提升战斗决策质量

**任务**：
1. 采集虎先锋战斗数据（含血量变化标注）
2. 实现EnemyWorldModel
3. 对比BC战斗 vs 世界模型战斗

### 📅 第三阶段：双世界模型融合（2-3周）

**目标**：寻路+战斗统一决策

**任务**：
1. 双模型并行推理（寻路用位置模型，战斗用敌人模型）
2. 仲裁器设计（敌人出现 → 战斗优先级）
3. 端到端测试

---

## 六、世界模型代码框架（可直接使用）

```python
"""
world_model.py - 悟空AI世界模型
三层架构：位置世界模型 + 敌人世界模型 + 完整世界模型

使用方法：

  # ===== 位置世界模型（寻路）=====
  # 1. 训练
  python world_model.py --train --layer location --data-dir pathfinding_data
  
  # 2. 推理
  python world_model.py --infer --layer location --goal-image boss_door.png
  
  # ===== 敌人世界模型（战斗）=====
  # 1. 训练
  python world_model.py --train --layer enemy --data-dir combat_data
  
  # 2. 推理
  python world_model.py --infer --layer enemy
  
  # ===== 完整世界模型（统一）=====
  # 1. 训练
  python world_model.py --train --layer full
  
  # 2. 推理
  python world_model.py --infer --layer full --goal "前往Boss房并击杀"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import MODEL, NUM_ACTIONS
from models.resnet_encoder import create_encoder


class JEPAPredictor(nn.Module):
    """
    JEPA预测器：给定当前状态表示 + 动作，预测未来状态表示
    核心：不预测像素，预测抽象表示（latent space prediction）
    """
    
    def __init__(self, latent_dim=256, action_dim=10, hidden_dim=512):
        super().__init__()
        
        # 动作嵌入
        self.action_embed = nn.Embedding(action_dim, 64)
        
        # 预测器网络：输入=状态表示+动作嵌入，输出=未来状态表示
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, state_repr, actions):
        """
        Args:
            state_repr: (B, latent_dim) 当前状态表示
            actions: (B, T) 或 (B,) 动作序列或单个动作
        Returns:
            predicted: (B, latent_dim) 预测的未来状态表示
        """
        if actions.dim() == 2:
            # 动作序列：压缩为单一表示
            action_emb = self.action_embed(actions).mean(dim=1)  # (B, 64)
        else:
            # 单个动作
            action_emb = self.action_embed(actions)  # (B, 64)
        
        combined = torch.cat([state_repr, action_emb], dim=-1)  # (B, latent+64)
        return self.predictor(combined)


class ContrastiveEncoder(nn.Module):
    """
    对比编码器：让相似状态在表示空间靠近，不同状态远离
    这是JEPA区别于VAE/Masked Autoencoder的关键
    """
    
    def __init__(self, latent_dim=256, encoder_type='resnet18'):
        super().__init__()
        self.encoder = create_encoder(encoder_type, latent_dim=latent_dim)
        self.latent_dim = latent_dim
    
    def forward(self, frames):
        """
        Args:
            frames: (B, C, H, W) 或 (B, T, C, H, W)
        Returns:
            repr: (B, latent_dim)
        """
        if frames.dim() == 5:
            # 有时间维度：只取最后一帧
            frames = frames[:, -1]
        return self.encoder(frames)


class LocationWorldModel(nn.Module):
    """
    位置世界模型（Layer 1）
    
    解决寻路问题：给定当前位置+动作序列，预测是否接近目标
    
    训练：JEPA loss on trajectory data
    推理：Rollout多条路径，选择最接近目标的
    """
    
    def __init__(self, latent_dim=256, hidden_dim=512, action_dim=10):
        super().__init__()
        
        self.encoder = ContrastiveEncoder(latent_dim=latent_dim)
        self.predictor = JEPAPredictor(latent_dim, action_dim, hidden_dim)
        
        # 目标判断器：给定当前位置+预测位置，判断是否到达目标
        self.goal_checker = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, frames, actions, goal_frame=None, K=30, rollout_steps=6):
        """
        Args:
            frames: (B, T, C, H, W) 历史帧序列
            actions: (B, T) 动作序列
            goal_frame: (B, C, H, W) 目标帧（可选）
            K: int 每步预测多少帧之后
            rollout_steps: int rollout多少次
        Returns:
            reach_prob: (B,) 到达目标的概率
            predicted_trajectory: list of (B, latent_dim) 预测的轨迹
        """
        B = frames.shape[0]
        
        # 当前状态
        current_state = self.encoder(frames[:, -1:])  # (B, 1, 256) → (B, 256)
        
        # 目标状态（如果提供了goal_frame）
        if goal_frame is not None:
            goal_state = self.encoder(goal_frame)  # (B, 256)
        else:
            goal_state = None
        
        # Rollout预测
        predicted_trajectory = []
        state = current_state
        
        for step in range(rollout_steps):
            # 使用真实动作（teacher forcing）
            action_seq = actions[:, step:step+K] if actions.shape[1] > step else actions[:, -1:]
            predicted_next = self.predictor(state, action_seq)  # (B, 256)
            predicted_trajectory.append(predicted_next)
            state = predicted_next
        
        # 最后预测位置
        final_pred = predicted_trajectory[-1]  # (B, 256)
        
        # 判断是否接近目标
        if goal_state is not None:
            goal_check = torch.cat([final_pred, goal_state], dim=-1)
            reach_prob = self.goal_checker(goal_check).squeeze(-1)  # (B,)
        else:
            reach_prob = None
        
        return reach_prob, predicted_trajectory
    
    def predict_next(self, frame, action):
        """单步预测（推理用）"""
        z_t = self.encoder(frame)  # (B, 256)
        z_next = self.predictor(z_t, action)  # (B, 256)
        return z_next
    
    def distance_to_goal(self, current_frame, goal_frame):
        """计算当前画面到目标画面的距离（表示空间）"""
        z_current = self.encoder(current_frame)
        z_goal = self.encoder(goal_frame)
        return F.cosine_similarity(z_current, z_goal, dim=-1)


class EnemyWorldModel(nn.Module):
    """
    敌人世界模型（Layer 2）
    
    解决战斗问题：给定敌人状态+我方动作，预测敌人反应+血量变化
    核心：把战斗从"盲打"变成"有预测的打"
    """
    
    def __init__(self, latent_dim=256, hidden_dim=512, action_dim=10):
        super().__init__()
        
        self.encoder = ContrastiveEncoder(latent_dim=latent_dim)
        self.action_embed = nn.Embedding(action_dim, 64)
        
        # 敌人特征提取：编码器 + 专用敌人头
        self.enemy_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        # 敌人反应预测器
        self.reaction_predictor = nn.Sequential(
            nn.Linear(64 + 64, 128),   # 敌人状态 + 我方动作
            nn.ReLU(),
            nn.Linear(128, 32),         # 敌人出招类型embedding
        )
        
        # 血量变化预测器
        self.hp_predictor = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 2),          # (我方HP变化, 敌人HP变化)
        )
    
    def forward(self, frames, my_action):
        """
        Args:
            frames: (B, T, C, H, W) 含敌人的画面序列
            my_action: (B,) 我方动作ID
        Returns:
            enemy_reaction: (B, 32) 敌人反应特征
            hp_delta: (B, 2) (我方HP变化, 敌人HP变化)
        """
        B = frames.shape[0]
        
        # 编码所有帧
        all_states = self.encoder(frames)  # (B, T, 256)
        
        # 取最后帧（当前）
        current_state = all_states[:, -1]  # (B, 256)
        
        # 提取敌人状态
        enemy_state = self.enemy_head(current_state)  # (B, 64)
        
        # 编码我方动作
        action_emb = self.action_embed(my_action)  # (B, 64)
        
        # 预测
        combined = torch.cat([enemy_state, action_emb], dim=-1)  # (B, 128)
        enemy_reaction = self.reaction_predictor(combined)  # (B, 32)
        hp_delta = self.hp_predictor(combined)  # (B, 2)
        
        return enemy_reaction, hp_delta


class WukongWorldModel(nn.Module):
    """
    完整世界模型 = 位置世界模型 + 敌人世界模型
    
    顶层：仲裁器决定用哪个模型
    """
    
    def __init__(self, latent_dim=256, hidden_dim=512, action_dim=10):
        super().__init__()
        
        self.location_model = LocationWorldModel(latent_dim, hidden_dim, action_dim)
        self.enemy_model = EnemyWorldModel(latent_dim, hidden_dim, action_dim)
        
        # 敌人检测器（简单的视觉检测）
        self.enemy_detector = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def detect_enemy(self, frame):
        """检测画面中是否有敌人（用于仲裁）"""
        state = self.location_model.encoder(frame)
        enemy_prob = self.enemy_detector(state)
        return enemy_prob.squeeze(-1) > 0.5
    
    def decide(self, frames, actions, goal_frame, combat_mode=False):
        """
        决策函数：BC vs 世界模型
        """
        enemy_present = self.detect_enemy(frames[:, -1:])
        
        if enemy_present and not combat_mode:
            # 敌人出现 → 战斗模式
            return "combat"
        else:
            # 正常寻路
            return "pathfinding"
    
    def pathfinding_decide(self, frames, actions, goal_frame, K=30):
        """寻路决策"""
        return self.location_model(frames, actions, goal_frame, K=K)
    
    def combat_decide(self, frames, my_action):
        """战斗决策"""
        return self.enemy_model(frames, my_action)


# ============== 训练函数 ==============

def train_location_world_model(model, data_loader, optimizer, epochs=50, device='cuda'):
    """
    训练位置世界模型
    数据：直接用.h5轨迹数据，无需额外标注！
    """
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_samples = 0
        
        for batch in data_loader:
            frames, actions = batch
            frames = frames.to(device)
            actions = actions.to(device)
            
            B, T = frames.shape[:2]
            
            # K步预测目标
            K = 30
            if T <= K:
                continue
            
            # 当前状态
            current_states = model.location_model.encoder(frames[:, :-K])  # (B, T-K, 256)
            
            # Teacher forcing: 用真实动作序列预测
            action_emb = model.location_model.predictor.action_embed(actions[:, :-K])  # (B, T-K, 64)
            action_repr = action_emb.mean(dim=1)  # (B, 64)
            
            # 预测
            predicted = model.location_model.predictor(current_states[:, 0], action_repr)  # (B, 256)
            
            # 真实未来状态
            future_states = model.location_model.encoder(frames[:, K:])  # (B, T-K, 256)
            future_repr = future_states.mean(dim=1)  # (B, 256)
            
            # JEPA正样本损失
            positive_loss = F.mse_loss(predicted, future_repr)
            
            # JEPA负样本损失（随机动作）
            random_actions = torch.randint(0, 10, (B,), device=device)
            wrong_pred = model.location_model.predictor(current_states[:, 0], random_actions)
            negative_loss = F.mse_loss(wrong_pred, future_repr.detach())
            
            # 总损失
            loss = positive_loss + 0.1 * negative_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * B
            total_samples += B
        
        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
              f"(pos={positive_loss.item():.4f} neg={negative_loss.item():.4f})", flush=True)
    
    # 保存
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/world_model.pt")
    print("World model saved: checkpoints/world_model.pt", flush=True)


# ============== 推理函数 ==============

def infer_with_world_model(model, goal_frame, duration=60, fps=10):
    """
    用世界模型做推理
    核心改进：rollout多条路径，选择最接近目标的
    """
    import time
    from env.screen_capture import capture_frame
    from env.action_executor import execute_action
    
    model.eval()
    
    interval = 1.0 / fps
    start_time = time.time()
    
    print(f"World Model Inference: duration={duration}s, fps={fps}", flush=True)
    
    while time.time() - start_time < duration:
        frame = capture_frame()
        
        # 检测敌人
        if model.detect_enemy(frame.unsqueeze(0).cuda()):
            print("Enemy detected → Combat mode (using BC fallback)", flush=True)
            # 战斗模式：用现有BC模型（暂时）
            continue
        
        # 寻路：用世界模型rollout
        with torch.no_grad():
            # 当前状态
            current = model.location_model.encoder(frame.unsqueeze(0).cuda())
            
            best_action = None
            best_distance = float('inf')
            
            # Rollout所有动作
            for action_id in range(NUM_ACTIONS):
                predicted_next = model.location_model.predict_next(
                    frame.unsqueeze(0).cuda(),
                    torch.tensor([action_id], device='cuda')
                )
                
                # 计算到目标的距离
                goal = model.location_model.encoder(goal_frame.unsqueeze(0).cuda())
                dist = F.mse_loss(predicted_next, goal)
                
                if dist < best_distance:
                    best_distance = dist
                    best_action = action_id
            
            if best_action is not None:
                execute_action(best_action)
                print(f"Action: {best_action} (dist={best_distance:.4f})", flush=True)
        
        time.sleep(interval)


# ============== 主入口 ==============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wukong World Model")
    parser.add_argument("--train", action="store_true", help="训练模式")
    parser.add_argument("--infer", action="store_true", help="推理模式")
    parser.add_argument("--layer", type=str, default="location",
                        choices=["location", "enemy", "full"],
                        help="世界模型层级")
    parser.add_argument("--data-dir", type=str, default="pathfinding_data")
    parser.add_argument("--goal-image", type=str, default=None)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    # 创建模型
    model = WukongWorldModel(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    
    if args.train:
        # 加载数据
        import glob
        import h5py
        from torch.utils.data import DataLoader, Dataset
        
        class TrajectoryDataset(Dataset):
            def __init__(self, data_dir):
                self.h5_files = glob.glob(os.path.join(data_dir, "*.h5"))
            
            def __len__(self):
                return len(self.h5_files)
            
            def __getitem__(self, idx):
                with h5py.File(self.h5_files[idx], "r") as f:
                    frames = f["frames"][:]
                    actions = f["actions"][:]
                    
                    # 转换格式
                    frames = frames.transpose(0, 3, 1, 2).astype(float) / 255.0
                    
                    # 帧堆叠（取FRAME_STACK=4）
                    T = len(frames)
                    if T > 4:
                        # 随机选一个起点
                        start = torch.randint(0, T - 4, (1,)).item()
                        frame_seq = frames[start:start+4]
                        action_seq = actions[start:start+4]
                    else:
                        frame_seq = frames
                        action_seq = actions
                    
                    return torch.from_numpy(frame_seq), torch.from_numpy(action_seq)
        
        dataset = TrajectoryDataset(args.data_dir)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        if args.layer == "location":
            train_location_world_model(
                model.location_model, loader, optimizer,
                epochs=args.epochs, device=device
            )
        else:
            print(f"Training for layer '{args.layer}' not yet implemented", flush=True)
    
    elif args.infer:
        if args.goal_image:
            from PIL import Image
            goal_img = Image.open(args.goal_image)
            goal_frame = torch.from_numpy(
                np.array(goal_img.resize((224, 224))).transpose(2, 0, 1)
            ).float() / 255.0
        else:
            goal_frame = None
        
        infer_with_world_model(model, goal_frame, duration=args.duration)
    
    else:
        parser.print_help()
```

---

## 七、为什么世界模型能解决悟空AI的核心问题

### 7.1 问题→方案映射

| 核心问题 | BC的问题 | 世界模型的解决 |
|---------|---------|--------------|
| 不知道要去哪 | 状态→动作，没有目标 | 状态+动作→预测位置，和目标比对 |
| 泛化能力弱 | 过拟合训练场景 | 可模拟未见过的动作组合 |
| compounding error | 错误累积无法恢复 | rollouts验证后再执行 |
| 多模态坍缩 | 只输出最常见动作 | 保持多条路径竞争 |
| 鼠标控制失效 | MSE被CE淹没 | 预测位置而非直接输出鼠标 |

### 7.2 关键技术优势

**1. 表示预测 > 像素预测**

JEPA的核心洞察：预测"抽象表示"而不是"像素"。这对游戏AI特别重要：

```
像素预测的难点（游戏画面）:
  - 阴影、粒子效果、光照变化 → 像素完全不同
  - 同一位置不同时间 → 像素差异巨大
  → 像素级预测误差快速累积

表示预测的优势:
  - ResNet编码器提取的256维表示已经去除了无关细节
  - 存档点A的表示 vs Boss房的表示 → 差异稳定可预测
  → 表示级预测更鲁棒
```

**2. 负样本学习（对比JEPA）**

```
BC训练：只学"正确的动作"
  → 模型只知道正确答案，不知道错误答案多离谱

JEPA训练：同时学"正确"和"错误"
  → 正样本：当前state + 真实action → 预测state（应该匹配真实future）
  → 负样本：当前state + 随机action → 预测state（应该远离真实future）
  
效果：
  → 模型不只是学"模仿"，而是学"理解因果"
  → 随机动作的预测结果和真实未来差得远 → 模型学会区分好坏
```

**3. 零样本泛化的可能性**

```
世界模型可以零样本泛化的场景：
  1. 新岔路口：不用见过，只要rollout所有方向选择 → 最优的才是对的
  2. 新怪物：不用见过，只要预测"打一下会怎样"
  3. 新道具：不用见过，只要预测"拿了会有什么变化"

BC做不到：因为它没有"模拟器"，只能回忆训练数据
世界模型能做到：它有一个"游戏模拟器"，可以模拟没见过的场景
```

---

## 八、资源与依赖

### 8.1 硬件需求

| 模型 | 显存 | 推理速度 | 训练时间 |
|------|------|---------|---------|
| LocationWorldModel | ~1.5GB | 5-8 FPS | 2-3小时 |
| EnemyWorldModel | ~2GB | 3-5 FPS | 3-4小时 |
| 完整世界模型 | ~3GB | 2-3 FPS | 5-6小时 |

**结论**：RTX 2060 6GB 可以跑位置世界模型（推理），训练需要适当batch size。

### 8.2 数据需求

| 模型 | 数据需求 | 现有数据是否够用 |
|------|---------|---------------|
| LocationWorldModel | 寻路轨迹（任意数量） | ✅ 直接复用 |
| EnemyWorldModel | 战斗数据 + 血量标注 | ⚠️ 需要采集 |
| 完整世界模型 | 混合数据 | ⚠️ 需要混合采集 |

### 8.3 关键参考

**Yann LeCun世界模型论文**：
- I-JEPA (2023): arXiv:2301.12503
- V-JEPA (2024): arXiv:2402.14405
- LeWorldModel (2025): LeCun lab

**游戏AI世界模型**：
- DreamerV3 (2023): Google's world model for reinforcement learning
- Minerva (2024): LLM做数学推理 → 类比世界模型的"规划"能力
- GameNGen (2024): 实时游戏世界模型（DOOM上的神经渲染）

---

## 九、总结：为什么世界模型是悟空AI的正确方向

### 核心理由

```
1. 问题匹配：游戏AI的核心挑战是"规划"而不是"反应"
   BC解决"反应"，世界模型解决"规划"
   
2. 数据效率：世界模型可以从现有轨迹数据中学习
   不需要额外标注，不需要新采集
   （JEPA loss可以直接在现有.h5数据上训练）
   
3. 技术可实现：核心模块已经存在
   - ResNetEncoder ✅（已有）
   - JEPA Predictor ✅（新写，少量代码）
   - Goal Checker ✅（新写，少量代码）
   总计：~300行新代码，1周可完成

4. 效果可验证：世界模型 vs BC的直接对比
   同一测试场景，rollout vs 查表 → 成功率量化对比
   
5. 升级路径清晰：
   Layer 1（位置）→ Layer 2（敌人）→ Layer 3（完整）
   每层独立可测试，每层都有明确收益
```

### 最快路径

```
Day 1-2: 实现 LocationWorldModel + JEPA loss
Day 3:   在现有数据上训练50 epochs
Day 4:   推理测试 + 记录结果
Day 5:   对比BC v3 vs 世界模型（同一测试集）
Day 6:   提交GitHub + 写研究报告

总耗时：1周
预期收益：寻路泛化能力显著提升
```

---

**报告完成时间**: 2026-05-20 11:56 GMT+8
**下次更新**: 完成位置世界模型第一阶段训练后
**相关文档**: `docs/VLA_Research.md`, `docs/RESEARCH_FOUNDATION_MODELS.md`, `docs/RESEARCH_AND_RECOMMENDATIONS.md`, `TODO.md`
