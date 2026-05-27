# wukong_ai 项目进展报告

**日期**: 2026-05-27  
**状态**: 行为克隆训练完成，推理测试进行中  
**核心问题**: 鼠标输入控制未完全解决

---

## ✅ 已完成工作

### 1. AI 关键人物传记系列（18本书）
- **已完成**: OpenClaw架构全书、Karpathy、DeepSeek、同量级公司、Aschenbrenner、LeCun、Sutskever、Hinton、Bengio、Amodei、Hassabis、Huang、Altman、Musk、Nadella、Zuckerberg、Schmidhuber、Pichai
- **进行中**: Sundar Pichai（资料已收集）
- **计划**: 吴恩达（第20本）
- **格式**: 标准12章结构，Markdown + PDF（reportlab + msyh.ttc）
- **位置**: `workspace/<name>-book/`

### 2. wukong_ai 项目（黑神话：悟空 AI）

#### 技术架构
- **算法**: Goal-Conditioned Behavior Cloning (GC-BC)
- **模型**: ResNet18 + Goal Embedding + 非均匀帧堆叠 [0, +1, +3, +7]
- **输入**: 4帧堆叠 (12通道 RGB) + goal_id (0/1)
- **输出**: 5类离散动作 (idle/forward/turn_slow/turn_medium/turn_fast) + 鼠标速度量化
- **训练数据**: 6719帧（人类采集）+ 1135帧（DAgger纠正）

#### 已完成实验
| 版本 | 方案 | 准确率 | 问题 |
|------|------|--------|------|
| v4.1 | 连续鼠标回归 | 63% | 协变量漂移，推理效果差 |
| v5 | 5类离散 + CE | 49-52% | 接近随机猜测 |
| v5.1 | 4帧堆叠 | 52.69% | 准确率停滞 |
| v5.2 | 非均匀帧堆叠 + Focal Loss | **94.06%** | 推理时鼠标控制无效 |

#### 当前 Checkpoint
- `goal_bc_v52_epoch_030.pt` (Epoch 30/30, Acc=94.29%)
- `goal_bc_v52_epoch_020.pt` (Epoch 20/30, Acc=93.75%)
- **⚠️ 未保存最优模型** (best_acc 只记录日志，未 torch.save)

---

## 🐛 已知问题

### 1. 鼠标输入控制（核心阻塞问题）
**问题描述**:
- ✅ SendInput API 有效（Windows光标动了）
- ❌ 游戏使用 Raw Input 读取鼠标，绕过 Windows 光标
- ❌ `pydirectinput` 无效（游戏不响应）
- ⚠️ 用户反馈："游戏画面有转向，但和鼠标操作有一点点不同"

**根因分析**:
- 黑神话（和所有现代游戏）使用 Raw Input API 直接读鼠标硬件
- SendInput / pydirectinput 只影响 Windows 光标，游戏收不到
- 游戏设置中**无 Raw Input 开关**（强制开启）

**可能的解决方案**:
1. **Hook Raw Input** - 用 C++ DLL 注入，直接模拟 Raw Input 消息
2. **驱动级模拟** - 使用 Interception 驱动（键盘鼠标驱动级拦截）
3. **视觉方案** - 不用鼠标事件，用 OpenCV 识别小地图方向
4. **游戏内存读取** - 读游戏坐标数据（需要逆向工程）

**参考项目**:
- [Turing-Project/Black-Myth-Wukong-AI](https://github.com/Turing-Project/Black-Myth-Wukong-AI) (392 stars)
  - 他们肯定解决了鼠标控制问题
  - 需要研究他们的代码

### 2. 数据分布失衡
**问题**: 
- idle + forward 占 87.6%，模型退化为常量预测
- 转向样本稀缺（right 7.5%, left 4.8%）

**已尝试方案**:
- ✅ 鼠标损失加权 (10x)
- ✅ 罕见动作加权 (5x)
- ✅ 起始帧鼠标损失加权 (20x)
- ✅ 方向一致性损失 (0.5x)
- ⚠️ 仍需更多转向数据

**建议**:
- 过滤 idle 帧（已创建 `filter_idle.py`）
- 采集更多转向数据（至少 20% 转向样本）
- 使用 DAgger 多轮迭代（当前仅 1 轮）

### 3. 协变量漂移（Behavior Cloning 根本缺陷）
**理论分析**:
- BC 假设：训练时专家演示覆盖所有状态
- 实际：模型错误累积，状态分布漂移
- 结果：推理时遇到训练未覆盖的状态，预测失败

**解决方案**:
1. **DAgger** (Dataset Aggregation) - 已尝试，但仅 1 轮迭代
2. **Goal-Conditioned BC + LSTM** - 时序建模，覆盖历史状态
3. **辅助任务** - 加重构损失、光流预测等

---

## 🔬 研究方向（待解决）

### 1. 鼠标输入控制（最高优先级）
**任务**: 研究如何让 AI 控制游戏鼠标（Raw Input）

**具体步骤**:
1. **研究 Turing 项目** - 克隆代码，分析鼠标控制实现
2. **测试 Raw Input Hook** - 用 C++ 写 DLL，注入游戏进程
3. **评估视觉方案** - 如果鼠标控制无法实现，改用 OpenCV

**预期时间**: 1-2 周

### 2. 数据质量提升
**任务**: 采集更多转向数据，平衡数据分布

**具体步骤**:
1. **运行 `filter_idle.py`** - 去掉 idle 帧，强制模型学转向
2. **DAgger 多轮迭代** - 至少 3-5 轮，直至干预率 < 25%
3. **数据增强** - 随机裁剪、颜色抖动、模拟不同光照

**预期时间**: 3-5 天

### 3. 模型架构改进
**任务**: 引入 LSTM / Transformer，提升时序建模能力

**具体步骤**:
1. **Goal-Conditioned BC + LSTM** - 覆盖历史状态，减少漂移
2. **辅助任务** - 加重建损失、光流预测
3. **Decision Transformer** - 离线强化学习（需要更多数据）

**预期时间**: 1-2 周

---

## 📊 下一步计划

### 短期（1-3天）
1. ✅ **提交当前进展到 GitHub** (本次提交)
2. ⏳ **研究 Turing 项目**，解决鼠标控制问题
3. ⏳ **过滤 idle 帧**，重新训练 v5.3
4. ⏳ **DAgger 第 2 轮迭代**，降低干预率

### 中期（1-2周）
1. ⏳ **解决鼠标输入问题**（Raw Input Hook / 视觉方案）
2. ⏳ **引入 LSTM**，提升时序建模
3. ⏳ **实机测试**，评估导航效果

### 长期（1个月+）
1. ⏳ **战斗 AI** - 目前只做导航，战斗还没开始
2. ⏳ **端到端方案** - 直接用 PPO/AI-GURKA 做 RL
3. ⏳ **多任务学习** - 导航 + 战斗 + 解谜

---

## 🤝 如何贡献

**欢迎贡献！** 以下是一些 Ideas：

### 代码贡献
- 修复 `goal_conditioned_bc_v52.py` 的 best_acc 保存 bug
- 实现 Raw Input Hook (C++ DLL)
- 添加视觉方案（OpenCV 识别小地图）
- 引入 LSTM / Transformer 架构

### 数据贡献
- 采集更多转向数据（goal 1 & 2）
- 录制不同地图、不同光照条件的数据
- 标注特殊场景（卡墙角、敌人出现、BOSS 战）

### 研究贡献
- 调研最新 BC/DAgger 改进方案
- 复现 Turing 项目的鼠标控制方法
- 设计更好的奖励函数（目前只有 goal 距离）

---

## 📞 联系方式

**Issues**: [GitHub Issues](https://github.com/Gravo/wukong_ai/issues)  
**Discussions**: [GitHub](https://github.com/Gravo/wukong_ai/discussions)  
**Email**: [待补充]

---

**最后更新**: 2026-05-27 00:08  
**维护者**: Gao Wei (Gravo)  
**License**: MIT
