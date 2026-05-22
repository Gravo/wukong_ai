# 周末研究计划 (2026-05-24 ~ 2026-05-25)

**目标**：验证核心方案，找到可行方向

---

## 原则

```
先验证，再扩展
跑通一个比规划十个更有价值
```

---

## 周六上午：验证 L2 辅助驾驶（09:00 - 12:00）

### 任务 1.1：测试规则方法
```bash
python assist/quick_start.py
```
- 预期：能看到自动闪避和自动面敌的输出
- 记录：规则方法的准确率

### 任务 1.2：采集闪避数据
```bash
python assist/data_collector.py --mode dodge --duration 300 --fps 15
```
- 预期：采集 300-500 个闪避样本
- 注意：专注于敌人攻击时按 Space

### 任务 1.3：训练闪避模型
```bash
python assist/train_dodge.py --data "l2_data/dodge_data_*.h5" --epochs 50
```
- 预期：验证准确率 > 60%
- 输出：checkpoints/auto_dodge_best.pt

---

## 周六下午：验证 BC v3（14:00 - 17:00）

### 任务 2.1：分析数据分布
```bash
python pathfinding/behavior_clone_v3.py --analyze-only
```
- 记录：当前 idle 占比

### 任务 2.2：训练 v3 模型
```bash
python pathfinding/behavior_clone_v3.py --epochs 100 --idle-ratio 0.1 --mouse-weight 2.0
```
- 预期：动作分布更均衡（idle < 20%）
- 输出：checkpoints/bc_v3_best.pt

### 任务 2.3：对比测试
```bash
# 打开游戏，进入虎先锋战斗
python pathfinding/inference_v2.py --duration 120 --fps 10
```
- 对比：v2 vs v3 的动作多样性
- 记录：鼠标输出是否有变化

---

## 周日上午：根据结果选择方向

### 决策矩阵

| L2 辅助 | BC v3 | 下一步 |
|---------|-------|--------|
| 有效 | 有效 | 优化 L2，实现自动连招 |
| 有效 | 无效 | 深入 L2，探索 MoE |
| 无效 | 有效 | 优化 BC v3，尝试 DAgger |
| 无效 | 无效 | 实现 DAgger 或 Goal-BC |

---

## 周日下午：深入实现（14:00 - 18:00）

### 如果选择 DAgger

```bash
# 1. 用当前模型控制角色
python pathfinding/inference_v2.py --duration 300

# 2. 同时运行数据采集，人类纠正
python training/data_collector.py --dagger

# 3. 用纠正数据重新训练
python pathfinding/behavior_clone_v3.py --dagger-data dagger_ep1.h5
```

### 如果选择 Goal-BC

```bash
# 1. 采集带 goal 标注的数据
python training/data_collector_v3.py --goal-mode

# 2. 训练 goal-conditioned 模型
python training/goal_conditioned_bc.py

# 3. 测试
python pathfinding/inference_v2.py --goal boss_door
```

---

## 任务清单

- [ ] 任务 1.1：测试 L2 规则方法
- [ ] 任务 1.2：采集闪避数据
- [ ] 任务 1.3：训练闪避模型
- [ ] 任务 2.1：分析数据分布
- [ ] 任务 2.2：训练 BC v3 模型
- [ ] 任务 2.3：对比测试
- [ ] 决策：选择下一步方向
- [ ] 任务 3：深入实现选定方案

---

## 关键文件

| 文件 | 用途 |
|------|------|
| `assist/quick_start.py` | L2 辅助驾驶快速启动 |
| `assist/data_collector.py` | 闪避数据采集 |
| `assist/train_dodge.py` | 闪避模型训练 |
| `pathfinding/behavior_clone_v3.py` | BC v3 训练 |
| `pathfinding/inference_v2.py` | 推理测试 |
| `training/goal_conditioned_bc.py` | Goal-BC 训练 |

---

## 记录模板

### 实验记录

```
日期：
任务：
配置：
结果：
观察：
下一步：
```

### 决策记录

```
时间点：
选项：
依据：
选择：
理由：
```

---

*计划生成时间: 2026-05-22*
