# 寻路问题辩论：不同视角的讨论

**日期**: 2026-05-19

---

## 辩论背景

本文档记录了关于黑神话悟空寻路问题定义的不同视角和讨论。通过对比不同观点，帮助理解问题的本质和最佳解决方案。

---

## 视角 A：纯工程视角（当前方案）

**代表观点**：行为克隆是最快、最简单的解决方案

### 核心论点

1. **简单直接**
   - 输入：游戏画面
   - 输出：动作
   - 不需要理解游戏内部机制

2. **端到端学习**
   - 不需要手工设计特征
   - 让模型自己学习什么是重要的

3. **快速迭代**
   - 录制数据 → 训练 → 测试
   - 可以快速验证想法

### 反驳

**问题 1**：为什么不学目标？

> 行为克隆只学 state→action 的映射，不理解动作的后果。模型不知道"要去哪"，只知道"这时候该按什么键"。

**问题 2**：为什么鼠标控制失效？

> 鼠标 dx/dy 是连续值，但大部分帧鼠标不动。MSE 损失被大量零值淹没，模型学会输出接近 0 的值。

**问题 3**：为什么无法泛化？

> Distribution shift：模型只在专家访问的状态上训练，但推理时会访问新的状态。一旦偏离，错误累积。

---

## 视角 B：计算机视觉视角

**代表观点**：寻路问题本质上是视觉理解问题

### 核心论点

1. **视觉感知是基础**
   - 人类导航靠眼睛
   - AI 也应该从视觉出发

2. **语义理解是关键**
   - 不只看像素，要理解"这是什么"
   - 识别可通行区域、障碍物、目标

3. **预训练模型可用**
   - CLIP、DINOv2 等模型已经学会了视觉语义
   - 可以直接利用这些知识

### 方案

```python
# 视觉语义导航
class VisualSemanticNavigator:
    def __init__(self):
        self.clip = load_clip()
        self.traversability = load_traversability_model()

    def navigate(self, image, goal_text):
        # 1. 语义理解
        semantic = self.clip.encode(image)

        # 2. 可通行性分析
        traversable = self.traversability.predict(image)

        # 3. 目标对齐
        goal_embedding = self.clip.encode_text(goal_text)

        # 4. 规划动作
        action = self.policy(semantic, traversable, goal_embedding)
        return action
```

### 反驳

**问题 1**：如何获取语义标注？

> 需要大量标注数据（"这是土地庙"、"这是可通行区域"）。手动标注成本高。

**问题 2**：实时性如何保证？

> CLIP 等模型推理较慢，可能无法满足实时控制需求。

**问题 3**：如何处理动态变化？

> 游戏画面会动态变化（敌人移动、光照变化），语义理解可能不稳定。

---

## 视角 C：强化学习视角

**代表观点**：寻路问题是序贯决策问题，应该用 RL 解决

### 核心论点

1. **理解后果**
   - RL 学习 state→action→reward 的映射
   - 模型知道动作的长期影响

2. **可以探索**
   - RL 可以尝试新策略
   - 不局限于专家数据

3. **分层规划**
   - 高层策略选择子目标
   - 低层策略执行动作

### 方案

```python
# 分层 RL 寻路
class HierarchicalRLNavigator:
    def __init__(self):
        self.high_level = HighLevelPolicy()  # 选择子目标
        self.low_level = LowLevelPolicy()    # 执行动作

    def navigate(self, observation, goal):
        # 高层：选择子目标
        subgoal = self.high_level(observation, goal)

        # 低层：执行动作到达子目标
        for _ in range(subgoal_horizon):
            action = self.low_level(observation, subgoal)
            observation, reward, done = env.step(action)

            if reached(observation, subgoal):
                break
```

### 反驳

**问题 1**：奖励如何设计？

> 寻路任务的奖励稀疏（只有到达目标才有正奖励）。需要精心设计 shaped reward。

**问题 2**：训练效率如何？

> RL 需要大量交互数据，训练效率低。

**问题 3**：如何保证安全？

> RL 可能探索到危险区域（如 Boss 攻击范围），导致训练不稳定。

---

## 视角 D：机器人学视角

**代表观点**：寻路问题是移动机器人导航问题，应该借鉴机器人学方法

### 核心论点

1. **成熟的方法论**
   - SLAM（同时定位与地图构建）
   - 路径规划算法（A*、RRT）
   - 避障算法（VFH、DWA）

2. **分层架构**
   - 感知层：理解环境
   - 规划层：规划路径
   - 控制层：执行动作

3. **可解释性**
   - 每一步决策都有明确原因
   - 便于调试和改进

### 方案

```python
# 机器人学风格的导航系统
class RobotStyleNavigator:
    def __init__(self):
        self.perception = PerceptionModule()  # 感知
        self.mapping = MappingModule()        # 建图
        self.planning = PlanningModule()      # 规划
        self.control = ControlModule()        # 控制

    def navigate(self, observation):
        # 1. 感知：提取环境信息
        features = self.perception(observation)

        # 2. 建图：更新地图
        self.mapping.update(features)

        # 3. 规划：找到路径
        path = self.planning.plan(self.mapping.get_map())

        # 4. 控制：执行动作
        action = self.control.follow_path(path)
        return action
```

### 反驳

**问题 1**：游戏环境不同于现实

> 游戏有固定的地图结构，不需要实时建图。而且游戏没有精确的坐标系统。

**问题 2**：如何处理战斗？

> 游戏中的导航和战斗是交织的，不能分开处理。

**问题 3**：实现复杂度

> 需要实现多个模块，每个模块都需要调优。

---

## 视角 E：游戏 AI 视角

**代表观点**：应该借鉴游戏 AI 的成功经验

### 核心论点

1. **游戏特定知识**
   - 游戏有固定的地图结构
   - 可以利用游戏机制（如传送点）

2. **混合方法**
   - 视觉 + 规则
   - 学习 + 搜索

3. **实用主义**
   - 不追求完美，追求可用
   - 快速迭代，逐步改进

### 方案

```python
# 游戏 AI 风格的导航
class GameAINavigator:
    def __init__(self):
        self.known_locations = {}  # 已知位置
        self.current_location = None
        self.goal = None

    def navigate(self, observation):
        # 1. 识别当前位置
        self.current_location = self.localize(observation)

        # 2. 如果到达目标，停止
        if self.current_location == self.goal:
            return "idle"

        # 3. 否则，找路径
        path = self.find_path(self.current_location, self.goal)

        # 4. 执行第一步
        return path[0]

    def localize(self, observation):
        """通过视觉特征识别位置"""
        # 使用预训练模型提取特征
        features = self.extract_features(observation)

        # 与已知位置匹配
        best_match = None
        best_score = -1
        for loc, feat in self.known_locations.items():
            score = cosine_similarity(features, feat)
            if score > best_score:
                best_score = score
                best_match = loc

        return best_match

    def find_path(self, start, goal):
        """在已知地图上找路径"""
        # 使用 A* 算法
        return astar(self.map, start, goal)
```

### 反驳

**问题 1**：如何获取游戏知识？

> 需要手动构建地图和规则，工作量大。

**问题 2**：如何处理新场景？

> 如果遇到未知区域，系统无法处理。

**问题 3**：如何保证泛化？

> 游戏更新或换一个游戏，系统可能失效。

---

## 综合讨论

### 各视角的优势

| 视角 | 优势 | 适用场景 |
|------|------|----------|
| 工程视角 | 简单、快速 | 原型验证、短期目标 |
| CV 视角 | 语义理解、泛化 | 探索、找目标 |
| RL 视角 | 长期规划、优化 | 复杂任务、长期目标 |
| 机器人视角 | 成熟、可靠 | 稳定导航、避障 |
| 游戏 AI 视角 | 实用、高效 | 特定游戏、快速部署 |

### 各视角的劣势

| 视角 | 劣势 | 局限 |
|------|------|------|
| 工程视角 | 不理解目标 | 无法泛化 |
| CV 视角 | 需要标注 | 实时性差 |
| RL 视角 | 训练复杂 | 奖励设计难 |
| 机器人视角 | 实现复杂 | 游戏不适用 |
| 游戏 AI 视角 | 依赖规则 | 泛化差 |

---

## 推荐方案：混合方法

### 核心思想

**结合各视角的优势，避免各自的劣势**

### 具体方案

```python
# 混合导航系统
class HybridNavigator:
    def __init__(self):
        # 视觉感知（CV 视角）
        self.visual_encoder = VisualEncoder()

        # 目标条件（工程视角）
        self.goal_encoder = GoalEncoder()

        # 位置识别（游戏 AI 视角）
        self.localizer = VisualLocalizer()

        # 路径规划（机器人视角）
        self.planner = AStarPlanner()

        # 动作执行（RL 视角）
        self.policy = PolicyNetwork()

    def navigate(self, observation, goal):
        # 1. 视觉感知
        visual_feat = self.visual_encoder(observation)

        # 2. 位置识别
        current_pos = self.localizer.localize(visual_feat)

        # 3. 路径规划
        path = self.planner.plan(current_pos, goal)

        # 4. 动作执行
        if path:
            next_pos = path[0]
            action = self.policy(visual_feat, next_pos)
        else:
            action = "idle"

        return action
```

### 实施路径

| 阶段 | 目标 | 方法 | 时间 |
|------|------|------|------|
| 1 | 改进 BC | 数据过滤 + 目标条件 | 1-2 周 |
| 2 | 位置识别 | 视觉特征匹配 | 2-3 周 |
| 3 | 地图构建 | 自动探索 + 人工标注 | 3-4 周 |
| 4 | 路径规划 | A* + 避障 | 4-5 周 |
| 5 | 端到端优化 | 分层 RL | 5-8 周 |

---

## 结论

### 问题定义的本质

**寻路问题 = 感知 + 规划 + 执行**

- **感知**：理解当前环境（视觉、语义）
- **规划**：决定去哪（目标、路径）
- **执行**：怎么去（动作、控制）

### 当前方案的问题

**只做了"执行"，没有做"感知"和"规划"**

### 解决方案

**分阶段实施，逐步添加"感知"和"规划"能力**

### 最终目标

构建一个**理解游戏语义、可以长期规划、能够泛化到新场景**的寻路系统。

---

*辩论记录时间: 2026-05-19*
