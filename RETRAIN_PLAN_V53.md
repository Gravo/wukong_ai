# wukong_ai 重新训练计划 - v5.3

**日期**: 2026-05-27  
**目标**: 解决当前模型问题，提升推理效果  
**计划执行**: 明天一早开始

---

## 🎯 核心要求（用户明确表达）

### 1. 转向需要有方向 🧭
**当前问题**:
- 模型预测 `pred_class in (2, 3, 4)` 只知道"要转向"
- 但**不知道转向方向**（左转还是右转）
- `last_mouse_dx` 不可靠（可能过时）

**解决方案**:
- ✅ **输出转向方向**（新增 `turn_direction` 输出头）
- ✅ 或者**用连续值表示转向角度**（-1.0 到 +1.0）

### 2. 转向需要有角度 📐
**当前问题**:
- 只有 3 类离散动作：`turn_slow`, `turn_medium`, `turn_fast`
- 实际应该是一个**连续的转角**（如 15°, 30°, 45°）

**解决方案**:
- ✅ **回归头输出转角**（`mouse_dx` 回归，-500 到 +500）
- ✅ 或者用**更细粒度的离散化**（10 个类别代替 3 个）

### 3. 转向占比需要提升 📊
**当前数据分布**:
- idle: 33.7%
- forward: 54.0%
- right: 7.5%
- left: 4.8%
- dodge: 0%

**问题**: 转向样本太少（12.3%），模型学不到

**解决方案**:
- ✅ **过滤 idle 帧**（`filter_idle.py`）
- ✅ **DAgger 多轮迭代**（至少 3-5 轮）
- ✅ **过采样转向帧**（重复采样 right/left）

### 4. 过滤掉 idle ✅
**当前问题**:
- 87.6% 样本是 idle+forward
- 模型退化为常量预测（永远预测 idle/forward）

**解决方案**:
- ✅ **训练时过滤 idle**（`filter_idle.py` → `data_noidle/`）
- ✅ **推理时跳过 idle**（预测 idle 时执行上次动作）

### 5. forward 是可以学习的 ✅
**当前观察**:
- forward 动作简单（只要按住 W）
- 模型容易学会（准确率 > 95%）

**结论**: ✅ **保留 forward**，不需要过滤

### 6. 当前的 idle 不需要学习 ❌
**用户明确要求**: "idle 不需要学习"

**解决方案**:
- ✅ **从数据集中删除所有 idle 帧**
- ✅ 模型只学 forward / turn_slow / turn_medium / turn_fast
- ✅ 推理时：如果模型输出 idle，继续执行上次动作

---

## 🐛 已知问题（需要修复）

### 问题1: 鼠标输入控制无效 🖱️
**表现**:
- SendInput 有效（Windows 光标动了）
- 但游戏用 Raw Input，**游戏内无响应**

**根因**: 现代游戏强制 Raw Input，无法通过设置关闭

**改进方案**:
1. **研究 Turing 项目** - 他们肯定解决了
2. **Hook Raw Input** - C++ DLL 注入
3. **视觉方案** - OpenCV 识别小地图方向

**优先级**: 🔴 最高（阻塞推理测试）

### 问题2: 协变量漂移（BC 根本缺陷）📉
**理论分析**:
- BC 假设：训练时专家演示覆盖所有状态
- 实际：模型错误累积，遇到未覆盖状态

**改进方案**:
1. **DAgger 多轮迭代** - 至少 3-5 轮，直至干预率 < 25%
2. **LSTM 时序建模** - 覆盖历史状态
3. **辅助任务** - 加重建损失、光流预测

**优先级**: 🟡 中等（需要更多数据和时间）

### 问题3: 数据分布失衡 📊
**当前分布**:
- idle: 33.7%
- forward: 54.0%
- right: 7.5%
- left: 4.8%

**改进方案**:
1. **过滤 idle** - `filter_idle.py` ✅
2. **采集更多转向数据** - 至少 20% 转向样本
3. **DAgger 纠正** - 让模型覆盖错误状态

**优先级**: 🟡 中等（过滤 idle 后缓解）

### 问题4: 模型输出不连续 🎲
**当前问题**:
- 离散动作（5 类）导致**跳跃式转向**
- 真人操作是**平滑连续**的

**改进方案**:
1. **连续值输出** - 回归头输出 `mouse_dx` (-500 to +500)
2. **平滑约束** - 添加 L2 正则，惩罚大幅变化
3. **时序平滑** - LSTM / EMA

**优先级**: 🟢 低（先解决方向+角度问题）

---

## 🔧 改进方案（v5.3 设计）

### 方案A：连续转角输出（推荐）✅

**模型架构**:
```
Input: 4帧堆叠 (12通道) + goal_id (0/1)
  ↓
ResNet18 (pretrained) + AdaptiveAvgPool2d
  ↓
Feature Vector (512-dim)
  ↓
LSTM (optional, 128-hidden)
  ↓
┌─────────────────┬─────────────────┐
│ Action Head     │ Mouse Head      │
│ (3类分类)      │ (2个回归值)    │
│ - forward       │ - mouse_dx      │
│ - turn_left     │ - mouse_dy      │
│ - turn_right    │                 │
└─────────────────┴─────────────────┘
```

**损失函数**:
```python
# Action loss (CE)
action_loss = CrossEntropyLoss(weights=[1.0, 5.0, 5.0])

# Mouse loss (MSE)
mouse_loss = MSELoss()

# Total loss
total_loss = action_loss + 10.0 * mouse_loss
```

**优点**:
- ✅ 转向有方向（正负号表示左右）
- ✅ 转向有角度（连续值 -500 to +500）
- ✅ 更符合真人操作

**缺点**:
- ⚠️ 回归任务比分类任务难学
- ⚠️ 需要更多数据

---

### 方案B：细粒度离散化 🔢

**动作空间** (11 类):
```
0: idle (可选，如果不过滤)
1: forward
2: turn_left_small   (-50px)
3: turn_left_medium  (-150px)
4: turn_left_large   (-300px)
5: turn_right_small  (+50px)
6: turn_right_medium (+150px)
7: turn_right_large  (+300px)
8: dodge
9: idle_special
10: unkno
```

**优点**:
- ✅ 转向有方向（left/right 分开）
- ✅ 转向有角度（small/medium/large）
- ✅ 分类任务容易学

**缺点**:
- ⚠️ 需要更多标注（细粒度动作）
- ⚠️ 数据分布更稀疏

---

### 方案C：过滤 idle + DAgger 多轮 🔄

**步骤**:
1. **过滤 idle 帧** (`filter_idle.py`)
2. **重新训练 v5.2** (只用 forward + turn)
3. **DAgger 第 2 轮** (采集错误状态)
4. **DAgger 第 3-5 轮** (直至干预率 < 25%)

**优点**:
- ✅ 简单（不改模型架构）
- ✅ 数据质量提升

**缺点**:
- ⚠️ 仍用离散动作（无连续转角）
- ⚠️ DAgger 需要多轮采集

---

## 📋 明天执行计划（2026-05-27 早上）

### 任务1: 过滤 idle 帧（30 分钟）✅

**命令**:
```powershell
cd D:\projects\wukong_ai
C:\Python\python.exe filter_idle.py
```

**预期输出**:
- 读取 `data\*.h5` (6719 帧)
- 过滤后 `data_noidle\*.h5` (约 687 帧)
- 数据分布：forward 29.4%, right 17.8%, left 11.2%, ...

### 任务2: 用过滤后数据重新训练（2-3 小时）⏳

**方案A：连续转角输出（推荐）**

**修改 `goal_conditioned_bc_v52.py`**:
1. 修改模型输出头（添加 `mouse_dx` 回归）
2. 修改损失函数（CE + MSE）
3. 用 `data_noidle` 训练

**命令**:
```powershell
C:\Python\python.exe -u training\goal_conditioned_bc_v53.py ^
  --data-dir "D:\projects\wukong_ai\data_noidle" ^
  --output-dir "D:\projects\wukong_ai\checkpoints" ^
  --batch-size 4 ^
  --epochs 50 ^
  --lr 1e-4
```

**方案B：细粒度离散化（备选）**

**修改数据采集器**:
- 记录细粒度动作（11 类）
- 重新采集数据（或手动标注）

**不推荐**（需要重新采集数据）

**方案C：过滤 idle + DAgger（简单）**

**命令**:
```powershell
# 训练
C:\Python\python.exe -u training\goal_conditioned_bc_v52.py \
  --data-dir "D:\projects\wukong_ai\data_noidle" \
  --epochs 50

# DAgger 第2轮
C:\Python\python.exe dagger_collector.py --round 2
```

### 任务3: 修复鼠标输入问题（研究任务）🔬

**步骤**:
1. **克隆 Turing 项目**:
   ```powershell
   cd D:\projects\wukong_ai
   git clone https://github.com/Turing-Project/Black-Myth-Wukong-AI.git
   ```

2. **研究他们的鼠标控制代码**:
   - 找 `mouse_controller.py` 或类似文件
   - 看他们用的是什么库/方法

3. **如果找到方案**，修改 `inference_goal_v53.py`

### 任务4: 测试推理效果（等待鼠标问题解决）🧪

**命令**:
```powershell
C:\Python\python.exe -u training\inference_goal_v53.py ^
  --model checkpoints\goal_bc_v53_best.pt ^
  --goal-id 0 ^
  --duration 60
```

**观察**:
- 转向是否有方向？
- 转向角度是否合理？
- 整体导航是否流畅？

---

## 📊 预期结果

### 如果方案A成功（连续转角输出）✅
- ✅ 转向有方向（正负号）
- ✅ 转向有角度（连续值）
- ✅ 推理效果接近真人

### 如果方案A失败（回归太难学）⚠️
- ⚠️ 改用方案C（过滤 idle + DAgger）
- ⚠️ 或采集更多数据（至少 5000 转向帧）

### 如果鼠标输入问题未解决 ❌
- ❌ 无法测试推理
- ⚠️ 改用视觉方案（不依赖鼠标事件）

---

## 🤝 如何协助

**代码贡献**:
- 实现方案A（连续转角输出）
- 修复 `goal_conditioned_bc_v52.py` 的 best_acc 保存 bug
- 实现 Raw Input Hook (C++ DLL)

**数据贡献**:
- 采集更多转向数据（goal 1 & 2）
- 运行 `filter_idle.py`
- DAgger 第 2-5 轮采集

**研究贡献**:
- 研究 Turing 项目的鼠标控制
- 调研连续控制方案（DPPO, SAC）
- 设计更好的奖励函数

---

## 📞 联系方式

**Issues**: [GitHub Issues](https://github.com/Gravo/wukong_ai/issues)  
**Discussions**: [GitHub Discussions](https://github.com/Gravo/wukong_ai/discussions)

---

**最后更新**: 2026-05-27 00:15  
**维护者**: Gao Wei (Gravo)  
**License**: MIT
