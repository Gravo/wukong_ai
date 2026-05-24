# 技术分析文档 - wukong_ai 项目

> **目标读者**：有强化学习经验、跑通过Gym系列项目（连连看、贪食蛇、CartPole、飞翔的小鸟），但想真正理解"从跑通到理解"鸿沟的开发者。

---

## 📊 目录

1. [当前网络结构详解](#1-当前网络结构详解)
2. [行为克隆（BC）的致命缺陷](#2-行为克隆bc的致命缺陷)
3. [DAgger算法：解决BC的根本问题](#3-dagger算法解决bc的根本问题)
4. [强化学习方案对比](#4-强化学习方案对比)
5. [下一步实验计划](#5-下一步实验计划)
6. [附录：关键代码解析](#6-附录关键代码解析)

---

## 1. 当前网络结构详解

### 1.1 整体架构（ASCII图）

```
┌─────────────────────────────────────────────────────────────┐
│                    输入层（Input）                        │
│  ┌──────────────────┐      ┌──────────────────┐       │
│  │ 游戏画面 (224×224×3) │      │  Goal ID (1或2)    │       │
│  └─────────┬────────┘      └─────────┬────────┘       │
│            ↓                        ↓                    │
│  ┌──────────────────┐      ┌──────────────────┐       │
│  │   ResNet18        │      │  Goal Embedding  │       │
│  │   (冻结参数)      │      │   (可训练)        │       │
│  │   输出: 512维    │      │   输出: 512维    │       │
│  └─────────┬────────┘      └─────────┬────────┘       │
│            ↓                        ↓                    │
│           └───────────┬──────────────┘              │
│                       ↓                               │
│              ┌──────────────────┐                    │
│              │   拼接 (Concat)  │                    │
│              │   512 + 512 = 1024 │                  │
│              └─────────┬────────┘                    │
│                        ↓                               │
│              ┌──────────────────┐                    │
│              │   FC层 (1024→512)                   │
│              │   + ReLU + Dropout │                  │
│              └─────────┬────────┘                    │
│                        ↓                               │
│         ┌──────────────┬──────────────┐             │
│         ↓              ↓              │             │
│  ┌─────────┐    ┌─────────┐      │             │
│  │action_head│    │mouse_head│      │             │
│  │(10类分类) │    │(2维回归) │      │             │
│  └─────────┘    └─────────┘      │             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 逐层详解

#### **输入层**
```python
# 游戏画面：224×224×3 (RGB)
frame = cv2.resize(frame, (224, 224))  # 缩放到224×224
frame = frame.transpose(2, 0, 1)       # HWC → CHW
frame = (frame - mean) / std             # ImageNet归一化

# Goal ID：整数 (0, 1, 2, ...)
goal_id = torch.tensor([1])  # Goal 1
```

#### **ResNet18（视觉特征提取）**
```python
# 来自 goal_conditioned_bc.py
self.visual_encoder = models.resnet18(pretrained=True)
self.visual_encoder = nn.Sequential(*list(self.visual_encoder.children())[:-1])
# 输出: (batch_size, 512, 1, 1) → (batch_size, 512)
```

**关键点**：
- ResNet18在ImageNet上预训练，**参数冻结**（不更新）
- 只做**特征提取**，不参与训练
- 输出512维向量（视觉语义特征）

#### **Goal Embedding（目标嵌入）**
```python
self.goal_embed = nn.Embedding(num_goals, 512)
# 输出: (batch_size, 512)
```

**关键点**：
- 把离散的goal_id（0,1,2...）映射为512维连续向量
- **可训练**——让模型学会"Goal 1代表什么方向"
- 类似NLP里的word embedding

#### **拼接 + FC层**
```python
fused = torch.cat([visual_feat, goal_embed], dim=1)
# fused shape: (batch_size, 1024)

fused = self.fusion_fc(fused)
# fusion_fc: Linear(1024, 512) + ReLU + Dropout
```

**关键点**：
- 把"视觉特征"和"目标特征"融合
- FC层降维回512（减少计算量）
- Dropout防止过拟合

#### **输出头**
```python
# 动作分类头
self.action_head = nn.Linear(512, num_actions)  # num_actions=10
# 输出: (batch_size, 10) → softmax → 动作概率

# 鼠标回归头
self.mouse_head = nn.Linear(512, 2)  # dx, dy
# 输出: (batch_size, 2) → 直接作为鼠标移动量
```

**关键点**：
- `action_head`：**分类问题**（CrossEntropyLoss）
- `mouse_head`：**回归问题**（MSELoss）
- 两个头**联合训练**（总损失 = action_loss + 10.0 * mouse_loss）

### 1.3 训练目标（Loss函数）

```python
# 动作分类损失
action_loss = F.cross_entropy(action_logits, action_labels)

# 鼠标回归损失
mouse_loss = F.mse_loss(mouse_pred, mouse_target)

# 方向一致性损失（新增v3.0）
direction_loss = -torch.mean(
    F.cosine_similarity(mouse_pred[:-1], mouse_pred[1:], dim=1)
)

# 总损失
total_loss = action_loss + 10.0 * mouse_loss + 0.5 * direction_loss
```

**关键点**：
- `mouse_loss`权重=10.0（因为鼠标预测更难）
- `direction_loss`惩罚"鼠标方向频繁翻转"（减少抖动）
- 这是**多任务学习**（Multi-Task Learning）

### 1.4 推理过程

```python
# 推理脚本: inference_goal.py
frame = camera.get_latest_frame()          # 抓取游戏画面
input_tensor = preprocess_frame(frame)       # 预处理
action_logits, mouse_pred = model(input_tensor, goal_id)  # 前向传播
action_id = torch.argmax(action_logits, dim=1).item()  # 动作分类
mouse_dx, mouse_dy = mouse_pred[0].numpy()  # 鼠标预测
mouse_dx *= pixels_per_unit  # 缩放（默认50）
pydirectinput.moveRel(mouse_dx, mouse_dy)  # 执行鼠标移动
pydirectinput.keyDown(action_key)  # 执行键盘动作
```

**关键点**：
- **单帧决策**——每一帧独立判断，没有时序信息
- `pixels_per_unit`把模型输出缩放到实际像素移动量
- EMA平滑（`alpha=0.3`）减少抖动

---

## 2. 行为克隆（BC）的致命缺陷

### 2.1 什么是行为克隆？

**行为克隆（Behavioral Cloning）** = **监督学习**（Supervised Learning）

```
训练数据: (状态, 动作) = (游戏画面, 人类操作的按键/鼠标)
目标: 学习一个策略 π(a|s) 使得模型输出接近人类操作
损失函数: L = -log P(人类动作 | 当前画面)
```

**看起来很美好，但实际有3个致命缺陷**：

### 2.2 缺陷1：协变量漂移（Covariate Shift）

#### **问题描述**

```
训练时: 状态分布来自人类操作 (分布 P_human)
测试时: 状态分布来自模型操作 (分布 P_model)
如果 P_human ≠ P_model → 模型在未见过的状态下会犯错
```

#### **具体例子（黑神话悟空）**

```
人类玩的时候:
  起始点 → 看到门 → 按'w'往前走 → 到达门
状态分布: [起始点, 门, 路]  (都是"正常 gameplay")

模型玩的时候:
  起始点 → (模型犯错) → 走到墙角 → 卡住
状态分布: [起始点, 墙角, 卡住]  (训练数据里没有"卡住"的状态)

结果: 模型在"卡住"状态下完全不知道该干嘛 → 继续卡住
```

#### **数学表达**

```
训练数据分布: P_human(s)
测试数据分布: P_model(s)

如果 P_model(s) ≠ P_human(s) → 泛化失败
```

#### **为什么会这样？**

```
人类操作时: 状态转移是 s_t → s_{t+1} (按人类策略)
模型操作时: 状态转移是 s_t' → s_{t+1}' (按模型策略)

因为模型会犯错 → s_t' 可能根本不在训练数据里 → 模型输出乱七八糟
```

### 2.3 缺陷2：错误累积（Error Accumulation）

#### **问题描述**

```
时刻 t: 模型犯了一个小错 (比如多走了1像素)
时刻 t+1: 因为状态已经偏离训练分布 → 模型更容易犯错
时刻 t+2: 错误越来越大...
结果: 几帧之后，模型完全跑飞
```

#### **具体例子**

```
人类玩:
  t=0: 起始点，正确转向门的方向
  t=1: 继续往前走
  t=2: 到达门

模型玩 (BC训练):
  t=0: 起始点，模型输出"往前走" (小错，应该先转向)
  t=1: 因为没转向，现在面对的是"墙" (训练数据里没有这个状态)
  t=2: 模型在"墙"状态下输出乱七八糟 → 卡住
```

#### **数学表达**

```
设模型策略为 π_model, 人类策略为 π_human

时刻 t 的状态分布: P_model^t(s)

如果 π_model ≠ π_human → P_model^t(s) 会越来越偏离 P_human(s)
→ 时刻 t 越大，模型犯错概率越高
```

### 2.4 缺陷3：类别不平衡（Class Imbalance）

#### **问题描述**

```
在你采集的数据里:
  idle (不动): 33.7%
  forward (往前走): 54.0%
  right (右转): 7.5%
  left (左转): 4.8%
  dodge (闪避): 0%

结果: 模型学会"永远输出 forward" → 准确率 87.7%，但完全不会转向
```

#### **为什么会这样？**

```
交叉熵损失: L = -Σ y_i * log(p_i)

如果 87.7% 的样本都是 idle+forward:
  模型输出"永远forward" → 损失下降 87.7%
  → 模型"偷懒"，不学少数类 (right, left, dodge)
```

#### **解决方案（你已经在做）**

```
1. 数据过滤: 去掉 idle 帧 (idle filtered)
2. 损失加权: 对少数类 (right, left) 加大权重
3. 过采样: 重复少数类样本
```

**但！这些都只是"治标不治本"** —— 根本问题还是协变量漂移+错误累积。

### 2.5 为什么你之前做的项目没有这个问题？

| 项目 | 状态空间 | 动作空间 | 奖励 | 为什么BC能用？ |
|------|---------|---------|------|----------------|
| **连连看** | 盘面（离散） | 点两个位置 | 消除得分 | 状态空间小，奖励密集 |
| **贪食蛇** | 盘面+蛇身 | 上下左右 | 吃食物+10，撞墙-100 | 同上 |
| **CartPole** | 4维向量（位置、速度、角度、角速度） | 左/右 | 每帧存活+1，倒下-1 | **状态空间极低维**，奖励清晰 |
| **飞翔的小鸟** | 鸟的位置+管道位置 | 跳/不跳 | 穿过管道+1，撞管道-100 | 状态空间简单 |
| **黑神话悟空** | **224×224图像（15万维）** | **10个键盘动作+鼠标连续运动** | **到达目标+1000，其他时候0** | ❌ **状态空间太高维，奖励稀疏** |

**核心矛盾**：
- 连连看/CartPole：**低维状态空间 + 清晰奖励**
- 黑神话悟空：**高维视觉输入 + 稀疏奖励（到达目标才算成功）**

**BC的假设**：
```
训练数据分布 ≈ 测试数据分布
```

**在黑神话悟空里，这个假设完全不成立** —— 模型一但犯错，状态分布立刻偏离训练数据。

---

## 3. DAgger算法：解决BC的根本问题

### 3.1 什么是DAgger？

**DAgger（Dataset Aggregation）** = **迭代式数据收集 + 监督学习**

```
核心思想:
  1. 用当前模型收集数据 (让AI自己玩)
  2. 让人类标注"在模型犯错的状态下，正确动作是什么"
  3. 把新数据加入训练集
  4. 重新训练模型
  5. 重复 1-4，直到模型收敛
```

**为什么DAgger能解决BC的问题？**

```
BC: 训练数据只来自人类操作 → 测试时模型遇到新状态就懵
DAgger: 训练数据来自"模型操作 + 人类纠正" → 逐步覆盖模型会犯错的状
```

### 3.2 DAgger算法流程（伪代码）

```python
# 初始化
D = ∅  # 数据集
π = None  # 策略（模型）

# 第1轮：用人类数据训练初始模型
D_human = collect_human_data()  # 你现在已经做了
D = D ∪ D_human
π = train_BC(D)

# 第2-10轮：DAgger迭代
for i in range(10):
    # 1. 用当前模型收集数据
    D_model = collect_model_data(π)  # 让AI自己玩
    
    # 2. 人类标注"在模型犯错的状态下，正确动作是什么"
    D_corrected = human_annotate(D_model)  # 人在关键时刻介入
    
    # 3. 把新数据加入训练集
    D = D ∪ D_corrected
    
    # 4. 重新训练模型
    π = train_BC(D)
    
    # 5. 评估模型性能
    success_rate = evaluate(π)
    print(f"Round {i}: success_rate={success_rate}")
```

### 3.3 DAgger在wukong_ai里的具体实现

#### **第1步：修改数据采集器（data_collector_v3.py）**

```python
# 新增"AI模式"
python data_collector_v3.py --mode ai --model checkpoints/goal_bc_epoch_030.pt --goal-id 1
```

**工作流程**：
```
1. 加载训练好的模型
2. 推理: 模型输出动作 + 鼠标移动
3. 执行动作（让AI自己玩）
4. 人类监控: 如果模型犯错 → 按某个键（比如'Q'）介入
5. 介入时: 记录"模型犯错的状态" + "人类的纠正动作"
6. 保存到新文件: pathfinding_dagger_round2.h5
```

#### **第2步：人类标注（你现在已经有了）**

```
不需要额外标注！
因为"人类介入时的动作"就是正确标注
```

#### **第3步：重新训练**

```python
# 把 D_human 和 D_dagger 合并
D_all = D_human ∪ D_dagger_round1 ∪ D_dagger_round2 ∪ ...

# 重新训练
python training/goal_conditioned_bc.py --data-dir D_all --epochs 30
```

#### **第4步：评估**

```
在游戏里测试:
  - Goal 1: 从起始点到门口，成功率？
  - Goal 2: 从起始点到前关卡方向，成功率？
```

### 3.4 DAgger vs BC：对比

| 方面 | BC | DAgger |
|------|----|--------|
| **数据来源** | 只有人类操作 | 人类操作 + 模型操作 + 人类纠正 |
| **协变量漂移** | ❌ 严重 | ✅ 逐步缓解 |
| **错误累积** | ❌ 严重 | ✅ 逐步缓解 |
| **数据效率** | ✅ 高（只用人类数据） | ⚠️ 中（需要多轮迭代） |
| **训练时间** | ✅ 快（一次性训练） | ⚠️ 慢（多轮训练） |
| **理论上限** | ❌ 低（只能模仿） | ✅ 高（能覆盖更多状态） |

### 3.5 你的情况：要不要上DAgger？

**✅ 强烈推荐！原因：**

1. **你有数据采集器**（改改就能用）
2. **DAgger样本效率高**（不需要几百万帧）
3. **你能直观看到改进**（每轮补充数据后推理测试）
4. **BC已经达到瓶颈**（99.8%准确率，但推理效果差）

**预计效果**：
```
Round 1 (BC): 成功率 20%
Round 2 (DAgger): 成功率 40%
Round 3 (DAgger): 成功率 60%
Round 4-10 (DAgger): 成功率 80%+
```

---

## 4. 强化学习方案对比

如果你想要"真正自主的AI"（而不是模仿人类），需要上**强化学习）（RL）。

### 4.1 为什么BC不够？

```
BC: 只能模仿人类，不能发现新路径
RL: 通过试错学习，能发现人类没做过的操作
```

**例子**：
```
人类玩: 走左边那条路
BC模型: 也只能走左边那条路
RL模型: 可能发现右边那条路更近！
```

### 4.2 RL方案对比

#### **方案A：PPO（Proximal Policy Optimization）**

```
优点:
  - 样本效率高（比REINFORCE高10倍）
  - 稳定（不会因为一次更新就崩）
  - 容易调参（只有2个超参数：clip_range, value_coef）

缺点:
  - 需要大量数据（几百万帧）
  - 训练时间长（几天到几周）

适合: 连续动作空间（鼠标dx, dy）
```

#### **方案B：A3C（Asynchronous Advantage Actor-Critic）**

```
优点:
  - 并行采样（多个worker同时玩游戏）
  - 训练速度快（比PPO快3-5倍）

缺点:
  - 不稳定（异步更新可能导致策略崩溃）
  - 调参困难（学习率、workers数量、etc.）

适合: 离散动作空间（键盘动作）
```

#### **方案C：IMPALA（Importance Weighted Actor-Learner Architecture）**

```
优点:
  - 大规模并行（100+ workers）
  - 适合分布式训练

缺点:
  - 实现复杂（需要自定义的分布式框架）
  - 不均衡（V-trace可能导致偏差）

适合: 超大规模训练（Google级别的）
```

### 4.3 推荐方案：PPO（如果你要上RL）

**为什么选PPO？**

1. **稳定**（最不容易崩）
2. **样本效率高**（你只有4394帧数据，PPO最省数据）
3. **容易实现**（Stable-Baselines3里有现成的）
4. **学术界主流**（90%的连续控制任务用PPO）

**PPO在wukong_ai里的实现**：

```python
import gym
from stable_baselines3 import PPO

# 定义环境
class WukongEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Dict({
            "keyboard": gym.spaces.Discrete(10),
            "mouse": gym.spaces.Box(low=-1, high=1, shape=(2,))
        })
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(224, 224, 3))
    
    def step(self, action):
        # 执行动作
        execute_action(action)
        
        # 获取下一帧
        next_frame = camera.get_latest_frame()
        
        # 计算奖励
        reward = compute_reward()
        
        # 判断是否结束
        done = is_episode_finished()
        
        return next_frame, reward, done, {}
    
    def reset(self):
        # 重置游戏到起始点
        reset_game()
        return camera.get_latest_frame()

# 创建环境
env = WukongEnv()

# 创建PPO模型
model = PPO("CnnPolicy", env, verbose=1)

# 训练
model.learn(total_timesteps=1000000)

# 保存
model.save("ppo_wukong")
```

### 4.4 奖励函数设计（最关键的部分！）

**如果你要上RL，奖励函数设计是成败关键**。

#### **奖励函数v1.0（最简单）**

```python
def compute_reward_v1():
    if reached_goal():
        return 1000  # 到达目标，大额奖励
    else:
        return 0  # 其他时候0
```

**问题**：**奖励太稀疏**（几万帧才能拿到一次+1000）→ 学不到东西

#### **奖励函数v2.0（基于距离变化）**

```python
def compute_reward_v2():
    if reached_goal():
        return 1000
    
    # 计算距离变化
    dist_now = compute_distance_to_goal()
    dist_prev = compute_distance_to_goal(prev_frame)
    dist_change = dist_prev - dist_now
    
    if dist_change > 0:
        return 1  # 靠近目标，小额奖励
    elif dist_change < 0:
        return -1  # 远离目标，小额惩罚
    else:
        return 0  # 没动
```

**问题**：**需要准确计算距离**（你怎么知道"距离目标还有多远"？）

#### **奖励函数v3.0（混合）**

```python
def compute_reward_v3():
    reward = 0
    
    # 1. 到达目标
    if reached_goal():
        reward += 1000
    
    # 2. 靠近目标
    dist_change = dist_prev - dist_now
    reward += dist_change * 0.1
    
    # 3. 存活奖励
    reward += 0.01  # 每帧存活，小额奖励
    
    # 4. 惩罚卡墙
    if is_stuck():
        reward -= 10
    
    return reward
```

**推荐这个** —— 兼顾"稀疏奖励"和"密集奖励"。

### 4.5 RL训练时间估算

```
假设:
  - 每秒钟采集10帧 (--fps 10)
  - PPO需要100万帧
  - 100万帧 ÷ 10帧/秒 = 10万秒 ≈ 27小时

实际:
  - 需要多轮迭代 (至少10轮)
  - 总时间: 27小时 × 10 = 270小时 ≈ 11天
```

**结论**：RL训练时间太长，不推荐现在就上。

---

## 5. 下一步实验计划

### 5.1 阶段1：DAgger（1-2周）

```
目标: 把成功率从 20% 提升到 80%

步骤:
  1. 改 data_collector_v3.py 支持"AI模式 + 人工干预"
  2. 跑 Round 1 DAgger（让当前模型自己玩，人类纠正）
  3. 重新训练
  4. 推理测试
  5. 重复 2-4，直到成功率 > 80%
```

**预计时间**：
- 每轮DAgger：1小时（采集数据）+ 30分钟（训练）= 1.5小时
- 10轮DAgger：15小时

### 5.2 阶段2：RL微调（2-4周，可选）

```
目标: 让模型发现新路径（不只是模仿人类）

步骤:
  1. 用BC/DAgger预训练模型（阶段1的结果）
  2. 用PPO继续训练
  3. 设计奖励函数（基于距离变化）
  4. 训练100万帧
  5. 评估是否发现新路径
```

**预计时间**：
- 100万帧采集：27小时
- 训练时间：12小时（RTX 2060）
- 总时间：~2天/轮 × 10轮 = 20天

### 5.3 阶段3：实机测试（持续）

```
目标: 在真实游戏里测试，收集bad case

步骤:
  1. 在游戏里测试（不同起始点、不同目标）
  2. 记录失败案例
  3. 分析失败原因
  4. 针对性补充数据/调整奖励函数
  5. 重新训练
```

---

## 6. 附录：关键代码解析

### 6.1 goal_conditioned_bc.py 完整解析

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class GoalConditionedBC(nn.Module):
    def __init__(self, num_goals=3, num_actions=10):
        super(GoalConditionedBC, self).__init__()
        
        # 1. 视觉编码器 (ResNet18)
        self.visual_encoder = models.resnet18(pretrained=True)
        self.visual_encoder = nn.Sequential(*list(self.visual_encoder.children())[:-1])
        # 冻结参数
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        
        # 2. Goal Embedding
        self.goal_embed = nn.Embedding(num_goals, 512)
        
        # 3. 融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 4. 输出头
        self.action_head = nn.Linear(512, num_actions)
        self.mouse_head = nn.Linear(512, 2)
    
    def forward(self, frames, goal_ids):
        # frames: (batch_size, 3, 224, 224)
        # goal_ids: (batch_size,)
        
        # 1. 视觉特征提取
        visual_feat = self.visual_encoder(frames)  # (batch_size, 512, 1, 1)
        visual_feat = visual_feat.squeeze(-1).squeeze(-1)  # (batch_size, 512)
        
        # 2. Goal Embedding
        goal_embed = self.goal_embed(goal_ids)  # (batch_size, 512)
        
        # 3. 融合
        fused = torch.cat([visual_feat, goal_embed], dim=1)  # (batch_size, 1024)
        fused = self.fusion_fc(fused)  # (batch_size, 512)
        
        # 4. 输出
        action_logits = self.action_head(fused)  # (batch_size, 10)
        mouse_pred = self.mouse_head(fused)  # (batch_size, 2)
        
        return action_logits, mouse_pred

# 损失函数
def compute_loss(action_logits, action_labels, mouse_pred, mouse_target, is_start):
    # 1. 动作分类损失
    action_loss = F.cross_entropy(action_logits, action_labels)
    
    # 2. 鼠标回归损失 (支持逐样本加权)
    mouse_criterion = nn.MSELoss(reduction='none')
    mouse_loss = mouse_criterion(mouse_pred, mouse_target)
    mouse_loss = (mouse_loss * mouse_weights).mean()  # 加权
    
    # 3. 起始帧鼠标损失加权 (v3.0新增)
    start_mouse_loss = mouse_loss[is_start].mean() * 20.0
    
    # 4. 方向一致性损失 (v3.0新增)
    direction_loss = -torch.mean(
        F.cosine_similarity(mouse_pred[:-1], mouse_pred[1:], dim=1)
    )
    
    # 5. 总损失
    total_loss = action_loss + 10.0 * mouse_loss + 20.0 * start_mouse_loss + 0.5 * direction_loss
    
    return total_loss, action_loss, mouse_loss
```

### 6.2 inference_goal.py 完整解析

```python
import torch
import numpy as np
import pydirectinput as pdi
import dxcam
import argparse

class EMASmoother:
    """指数移动平均平滑器"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None
    
    def update(self, x):
        if self.value is None:
            self.value = x.copy()
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value.copy()

def preprocess_frame(frame):
    """预处理游戏画面"""
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    frame = (frame - mean) / std
    frame = frame.transpose(2, 0, 1)
    return torch.from_numpy(frame).unsqueeze(0)

def execute_action(action_id, mouse_dx, mouse_dy, pixels_per_unit):
    """执行动作"""
    # 鼠标移动
    dx = int(mouse_dx * pixels_per_unit)
    dy = int(mouse_dy * pixels_per_unit)
    if abs(dx) > 1 or abs(dy) > 1:
        pdi.moveRel(dx, dy, relative=True)
    
    # 键盘动作
    key = ACTION_KEYS.get(action_id)
    if key:
        pdi.keyDown(key)
        time.sleep(0.05)
        pdi.keyUp(key)

def main(args):
    # 1. 加载模型
    model = GoalConditionedBC(num_goals=args.num_goals)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    
    # 2. 初始化摄像头
    camera = dxcam.create(output_color="BGR", region=(0, 0, 1920, 1080))
    camera.start(region=(0, 0, 1920, 1080), target_fps=args.fps)
    
    # 3. 推理循环
    mouse_smoother = EMASmoother(alpha=args.ema_alpha)
    goal_id = torch.tensor([args.goal_id], dtype=torch.long)
    
    for frame in camera:
        # 预处理
        input_tensor = preprocess_frame(frame)
        
        # 推理
        with torch.no_grad():
            action_logits, mouse_pred = model(input_tensor, goal_id)
        
        # 后处理
        action_id = torch.argmax(action_logits, dim=1).item()
        mouse_dx, mouse_dy = mouse_pred[0].numpy()
        smoothed_mouse = mouse_smoother.update(np.array([mouse_dx, mouse_dy]))
        
        # 执行
        execute_action(action_id, smoothed_mouse[0], smoothed_mouse[1], args.pixels_per_unit)
```

### 6.3 关键超参数解释

| 超参数 | 值 | 作用 | 为什么选这个值？ |
|--------|------|------|----------------|
| `mouse_loss`权重 | 10.0 | 加大鼠标预测的重要性 | 鼠标预测比动作分类难 |
| `start_mouse`权重 | 20.0 | 强制学"开局先转向" | 解决"起始点不转向"问题 |
| `direction`权重 | 0.5 | 减少鼠标抖动 | 惩罚方向频繁翻转 |
| `ema_alpha` | 0.3 | 平滑鼠标输出 | 平衡"响应速度"和"稳定性" |
| `pixels_per_unit` | 50 | 缩放鼠标输出 | 让模型输出对应实际像素移动 |
| `dropout` | 0.5 | 防止过拟合 | 标准值 |
| `lr` | 0.001 | 学习率 | Adam优化器的标准值 |

---

## 7. 总结：从"跑通"到"理解"的鸿沟

### 7.1 你现在已经理解的

✅ **网络结构**：ResNet18做特征提取，Goal Embedding做目标条件，FC层融合，两个输出头
✅ **训练目标**：动作分类 + 鼠标回归 + 方向一致性
✅ **推理过程**：单帧决策，EMA平滑，pixels_per_unit缩放

### 7.2 你还需要理解的

❓ **为什么BC不够？** → 协变量漂移 + 错误累积 + 类别不平衡
❓ **DAgger为什么能解决？** → 迭代式数据收集，覆盖模型会犯错的状
❓ **RL要不要上？** → 现阶段不需要，DAgger够了
❓ **奖励函数怎么设计？** → 如果上RL，需要基于距离变化

### 7.3 下一步行动

1. **改data_collector_v3.py** → 支持AI模式 + 人工干预
2. **跑DAgger Round 1** → 让当前模型自己玩，人类纠正
3. **重新训练** → 把DAgger数据加入训练集
4. **推理测试** → 看成功率是否提升
5. **重复3-4** → 直到成功率 > 80%

---

## 8. 参考资料

1. **DAgger原始论文**：
   - *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (Ross et al., 2011)
   - https://arxiv.org/abs/1011.0686

2. **PPO论文**：
   - *Proximal Policy Optimization Algorithms* (Schulman et al., 2017)
   - https://arxiv.org/abs/1707.06347

3. **ResNet论文**：
   - *Deep Residual Learning for Image Recognition* (He et al., 2016)
   - https://arxiv.org/abs/1512.03385

4. **Stable-Baselines3文档**：
   - https://stable-baselines3.readthedocs.io/

---

**文档版本**: v1.0
**最后更新**: 2026-05-24
**作者**: wukong_ai team
