# 黑神话：悟空 寻路问题定义与研究

**研究日期**: 2026-05-19

---

## 一、问题背景

### 1.1 为什么需要 AI 寻路？

《黑神话：悟空》是一款半开放世界的动作 RPG，玩家需要在复杂的 3D 环境中导航。游戏的特点：

- **无小地图**：设计意图是鼓励玩家自主探索和记忆路线
- **复杂地形**：山林、洞穴、寺庙等多层次立体空间
- **隐藏路径**：大量隐藏道路、秘密区域和支线任务
- **动态障碍**：敌人巡逻、环境陷阱、可破坏物体

### 1.2 当前方案的局限

当前的行为克隆方案存在根本性问题：

| 问题 | 原因 | 影响 |
|------|------|------|
| 只学动作，不学目标 | BC 只模仿 state→action | 模型不知道"要去哪" |
| 鼠标控制失效 | 连续值预测 + 稀疏奖励 | 视角控制几乎无效 |
| 无法处理新场景 | 分布偏移 | 偏离训练轨迹就失效 |
| 缺乏长期规划 | 无记忆机制 | 无法规划路径 |

---

## 二、寻路问题的多种定义

### 定义 1：纯视觉行为克隆（当前方案）

**问题定义**：
```
输入: 游戏画面 (224x224x12)
输出: 动作 (WASD + 鼠标)
目标: 模仿人类玩家的操作
```

**优点**：
- 简单直接
- 不需要游戏内部信息
- 可以端到端训练

**缺点**：
- 不理解游戏语义
- 无法泛化到新场景
- 鼠标控制几乎失效
- 缺乏长期规划能力

**适用场景**：
- 短距离导航（如 Boss 战中的走位）
- 已知路线的重复执行
- 简单的直线移动

---

### 定义 2：视觉语义导航（Semantic Navigation）

**问题定义**：
```
输入: 游戏画面 + 目标描述（如"去土地庙"、"找隐藏宝箱"）
输出: 动作序列
目标: 到达指定语义位置
```

**核心思想**：
- 不只看像素，要理解画面中的"东西"
- 识别可通行区域、障碍物、目标物体
- 将寻路问题分解为：感知 → 规划 → 执行

**技术方案**：
1. **视觉感知**：用预训练的视觉模型（如 CLIP、DINOv2）提取语义特征
2. **可通行性判断**：学习哪些区域可以走，哪些是障碍
3. **目标条件导航**：给定目标，规划路径

**示例**：
```python
# 语义导航系统
class SemanticNavigator:
    def __init__(self):
        self.clip_model = load_clip()  # 语义理解
        self.traversability_model = load_traversability()  # 可通行性
        self.goal_encoder = GoalEncoder()  # 目标编码

    def navigate(self, observation, goal_text):
        # 1. 语义理解
        semantic_features = self.clip_model.encode(observation)

        # 2. 可通行性分析
        traversable_mask = self.traversability_model.predict(observation)

        # 3. 目标编码
        goal_embedding = self.goal_encoder.encode(goal_text)

        # 4. 规划路径
        action_plan = self.plan_path(semantic_features, traversable_mask, goal_embedding)

        return action_plan
```

**优点**：
- 可以处理新场景（语义泛化）
- 支持多种目标（文字描述）
- 更接近人类的导航方式

**缺点**：
- 需要大量标注数据
- 实现复杂
- 实时性要求高

---

### 定义 3：拓扑图导航（Topological Navigation）

**问题定义**：
```
输入: 当前位置（视觉）+ 目标节点
输出: 到下一个节点的动作
目标: 在拓扑图上找到最短路径
```

**核心思想**：
- 将游戏世界抽象为图（Graph）
- 节点 = 关键位置（土地庙、Boss 区域、岔路口）
- 边 = 连接路径
- 寻路 = 图搜索问题

**技术方案**：

```python
# 拓扑图导航系统
class TopologicalNavigator:
    def __init__(self):
        self.graph = {
            "shrine_1": {"fork_1": "forward"},
            "fork_1": {"shrine_1": "back", "boss_area": "left", "secret_area": "right"},
            "boss_area": {"fork_1": "back"},
            "secret_area": {"fork_1": "back"},
        }
        self.current_node = None
        self.goal_node = None

    def localize(self, observation):
        """通过视觉识别当前位置（哪个节点）"""
        # 使用视觉特征匹配已知位置
        return self.visual_localization(observation)

    def plan_path(self, current, goal):
        """在图上搜索最短路径"""
        from collections import deque
        queue = deque([(current, [])])
        visited = {current}

        while queue:
            node, path = queue.popleft()
            if node == goal:
                return path

            for neighbor, action in self.graph[node].items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [action]))

        return None  # 无法到达

    def navigate(self, observation, goal_node):
        current_node = self.localize(observation)
        if current_node == goal_node:
            return "idle"

        path = self.plan_path(current_node, goal_node)
        if path:
            return path[0]  # 执行第一步
        return None
```

**优点**：
- 高效的路径规划
- 可以处理复杂的分支结构
- 易于扩展和修改

**缺点**：
- 需要预先构建地图
- 位置识别可能不准确
- 无法处理动态障碍

---

### 定义 4：分层强化学习（Hierarchical RL）

**问题定义**：
```
高层策略: 选择子目标（如"去岔路口"、"躲避敌人"）
低层策略: 执行具体动作到达子目标
目标: 学习分层的导航策略
```

**核心思想**：
- 将复杂的导航任务分解为多个子任务
- 高层策略负责"去哪"，低层策略负责"怎么去"
- 类似人类的导航方式：先决定大方向，再处理细节

**技术方案**：

```python
# 分层导航系统
class HierarchicalNavigator:
    def __init__(self):
        self.high_level_policy = HighLevelPolicy()  # 选择子目标
        self.low_level_policy = LowLevelPolicy()    # 执行动作
        self.subgoal_horizon = 50  # 每 50 步重新选择子目标

    def navigate(self, observation, goal):
        # 高层策略：选择子目标
        subgoal = self.high_level_policy(observation, goal)

        # 低层策略：执行动作到达子目标
        for step in range(self.subgoal_horizon):
            action = self.low_level_policy(observation, subgoal)
            observation, reward, done = env.step(action)

            if self.reached_subgoal(observation, subgoal):
                break

            if done:
                return

        # 递归：继续下一个子目标
        self.navigate(observation, goal)
```

**优点**：
- 可以处理长期规划
- 学习效率高
- 可以复用子策略

**缺点**：
- 训练复杂
- 子目标选择可能不优
- 需要精心设计奖励函数

---

### 定义 5：地图增强导航（Map-Augmented Navigation）

**问题定义**：
```
输入: 游戏画面 + 内存地图（累积的探索信息）
输出: 动作
目标: 结合实时视觉和历史信息进行导航
```

**核心思想**：
- 构建一个"内存地图"，记录已探索的区域
- 结合实时视觉和历史信息进行决策
- 类似人类的"空间记忆"

**技术方案**：

```python
# 地图增强导航系统
class MapAugmentedNavigator:
    def __init__(self):
        self.memory_map = SpatialMemory()  # 空间记忆
        self.visual_encoder = VisualEncoder()  # 视觉编码
        self.policy = PolicyNetwork()  # 决策网络

    def navigate(self, observation):
        # 1. 更新空间记忆
        visual_features = self.visual_encoder(observation)
        self.memory_map.update(visual_features)

        # 2. 获取地图特征
        map_features = self.memory_map.get_features()

        # 3. 联合决策
        combined_features = torch.cat([visual_features, map_features], dim=-1)
        action = self.policy(combined_features)

        return action
```

**优点**：
- 可以利用历史信息
- 避免重复探索
- 更鲁棒的导航

**缺点**：
- 需要维护地图状态
- 地图更新可能有误差
- 计算开销较大

---

## 三、问题定义对比

| 定义 | 输入 | 输出 | 核心能力 | 适用场景 | 复杂度 |
|------|------|------|----------|----------|--------|
| 视觉 BC | 画面 | 动作 | 模仿 | 短距离、已知路线 | 低 |
| 语义导航 | 画面 + 目标 | 动作序列 | 语义理解 | 探索、找目标 | 高 |
| 拓扑图 | 画面 + 图 | 节点动作 | 路径规划 | 大范围导航 | 中 |
| 分层 RL | 画面 + 目标 | 动作 | 长期规划 | 复杂任务 | 很高 |
| 地图增强 | 画面 + 地图 | 动作 | 空间记忆 | 探索、避障 | 高 |

---

## 四、推荐方案：分阶段实施

### 阶段 1：改进的视觉 BC + 目标条件（1-2 周）

**目标**：在当前 BC 基础上添加"目标感知"

**方案**：
1. 将目标信息编码为额外输入（如"去 Boss 区域"）
2. 训练时标注每帧的目标状态
3. 模型学习：给定目标，如何到达

```python
# 目标条件 BC
class GoalConditionedBC(nn.Module):
    def __init__(self):
        self.visual_encoder = ResNet18()
        self.goal_encoder = nn.Embedding(num_goals, 32)
        self.policy = nn.Linear(256 + 32, num_actions)

    def forward(self, observation, goal_id):
        visual_feat = self.visual_encoder(observation)
        goal_feat = self.goal_encoder(goal_id)
        combined = torch.cat([visual_feat, goal_feat], dim=-1)
        return self.policy(combined)
```

### 阶段 2：拓扑图构建 + 位置识别（2-4 周）

**目标**：自动构建游戏地图并识别当前位置

**方案**：
1. **地图构建**：通过探索自动构建拓扑图
2. **位置识别**：用视觉特征匹配已知位置
3. **路径规划**：在图上搜索最短路径

**数据收集**：
- 录制多条从 A 到 B 的路径
- 自动提取关键帧（转弯、岔路口）
- 构建节点和边

### 阶段 3：分层 RL（4-8 周）

**目标**：学习长期规划能力

**方案**：
1. **高层策略**：选择子目标（用 RL 训练）
2. **低层策略**：到达子目标（用 BC 初始化）
3. **联合训练**：端到端优化

---

## 五、具体实施建议

### 5.1 短期：改进当前 BC

**立即可做**：
1. 使用 v3 训练（数据过滤 + LR 调度）
2. 添加目标条件（简单的 one-hot 编码）
3. 增加鼠标控制的损失权重

**代码改动**：
```python
# 在 behavior_clone_v3.py 中添加目标条件
class GoalConditionedDataset(FilteredH5Dataset):
    def __init__(self, h5_path, goal_id, **kwargs):
        super().__init__(h5_path, **kwargs)
        self.goal_id = goal_id

    def __getitem__(self, idx):
        frame, action, mouse = super().__getitem__(idx)
        goal = np.array([self.goal_id], dtype=np.float32)
        return frame, action, mouse, goal
```

### 5.2 中期：拓扑图导航

**需要实现**：
1. **位置识别模块**：用 CLIP 或 DINOv2 提取视觉特征
2. **地图构建模块**：自动发现和连接节点
3. **路径规划模块**：A* 或 Dijkstra 搜索

**数据需求**：
- 多条完整路线的录制
- 每条路线的起始和结束标签

### 5.3 长期：分层 RL

**需要实现**：
1. **高层策略网络**：选择子目标
2. **低层策略网络**：执行动作
3. **奖励设计**：到达子目标 + 最终目标

---

## 六、与当前方案的对比

### 当前方案（行为克隆）

```
问题: state → action
优点: 简单、端到端
缺点: 不理解目标、无法规划、泛化差
```

### 改进方案（目标条件导航）

```
问题: (state, goal) → action
优点: 理解目标、可以泛化、支持多种任务
缺点: 需要目标标注、实现复杂
```

### 最终方案（分层 RL）

```
问题: (state, goal) → high_level_goal → low_level_action
优点: 长期规划、高效学习、可复用
缺点: 训练复杂、奖励设计难
```

---

## 七、关键挑战

### 7.1 位置识别

**挑战**：如何识别当前在游戏中的位置？

**方案**：
1. **视觉特征匹配**：用预训练模型提取特征，与已知位置匹配
2. **里程计**：通过动作和视觉变化估计位移
3. **GPS 模拟**：如果能获取游戏内部坐标，直接使用

### 7.2 地图构建

**挑战**：如何自动构建游戏地图？

**方案**：
1. **探索式构建**：边探索边构建
2. **从视频学习**：从人类游玩视频中提取地图
3. **游戏数据挖掘**：如果能获取游戏地图数据

### 7.3 动态障碍

**挑战**：如何处理敌人、陷阱等动态障碍？

**方案**：
1. **实时避障**：用视觉检测障碍并绕行
2. **预测性规划**：预测敌人移动并规划路径
3. **战斗集成**：将战斗和导航统一为一个策略

---

## 八、总结

### 问题定义的核心

寻路问题的本质是：**给定目标，如何从当前位置到达目标位置？**

当前方案的问题是：**只学"怎么走"，不学"去哪走"**

### 推荐路径

| 阶段 | 目标 | 时间 | 难度 |
|------|------|------|------|
| 1 | 改进 BC + 目标条件 | 1-2 周 | 低 |
| 2 | 拓扑图导航 | 2-4 周 | 中 |
| 3 | 分层 RL | 4-8 周 | 高 |

### 最终目标

构建一个**理解游戏语义、可以长期规划、能够泛化到新场景**的寻路系统。

---

*研究生成时间: 2026-05-19*
