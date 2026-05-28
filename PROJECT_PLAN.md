# wukong_ai 项目规划 - gstack 工作框架

**日期**: 2026-05-27  
**基于**: 同行评审意见 + 用户决策  
**核心原则**: 先攻坚，后优化

---

## 🎯 当前项目状态

### 已完成
- ✅ Goal-Conditioned BC v5.2 训练完成（Acc=94.06%）
- ✅ 数据采集器 v3（修复 G 键逻辑）
- ✅ DAgger 采集器 + 1轮数据
- ✅ L2 辅助驾驶系统（AutoDodge/AutoFace/Arbitrator）
- ✅ PointNav 模型框架
- ✅ GitHub 仓库（commit 1adccb6）

### 核心阻塞
🔴 **鼠标控制完全失效** — 游戏使用 Raw Input，所有 Windows 级模拟（SendInput/pydirectinput/pyautogui）均无效。模型无法在游戏中执行任何操作，**项目完全卡死**。

### 非阻塞但重要
- 🟡 数据分布失衡（87.6% idle+forward）
- 🟡 BC 协变量漂移（根本性缺陷）
- 🟡 模型无转向方向（只有"转多快"，无"往哪转"）

---

## 📋 工作规划（gstack 框架）

### Phase 0: 输入攻坚（当前阶段，最高优先级）

**目标**: 找到至少一种能让鼠标在黑神话中生效的方案  
**时间**: 1-2 周  
**原则**: 暂停一切模型/数据优化，全部资源投入输入问题

#### Task 0.1: 研究 Turing 项目 [优先级 P0]
**预计**: 2-3 小时  
**状态**: 未开始

```powershell
cd D:\projects\wukong_ai
git clone https://github.com/Turing-Project/Black-Myth-Wukong-AI.git
```

**执行步骤**:
1. 克隆项目，阅读 README 和整体架构
2. 搜索鼠标控制代码（关键词：mouse, input, SendInput, RawInput, move, control）
3. 分析他们的方案（什么库、什么API、什么原理）
4. 复现到我们的项目中
5. 在游戏中测试是否有效

**完成标准**: 明确 Turing 项目用的是什么输入方案，能否复用

#### Task 0.2: 搜索其他黑神话 AI 项目 [优先级 P0]
**预计**: 1-2 小时  
**状态**: 未开始

**搜索关键词**:
- "black myth wukong AI bot"
- "黑神话 自动脚本"
- "game AI mouse input raw input"
- "SendInput Raw Input game control"

**目标**: 找到 3+ 个类似项目，对比他们的输入方案

#### Task 0.3: Raw Input Hook 研发 [优先级 P1]
**预计**: 1-2 天  
**状态**: 未开始  
**前置**: Task 0.1 确认 Turing 项目不是 Hook 方案

**方案**: 用 C++ DLL 注入，Hook `GetRawInputData`，模拟 RAWINPUT 消息

**执行步骤**:
1. 研究 Windows Raw Input API（WM_INPUT, RAWINPUT struct）
2. 写 C++ DLL（Hook GetRawInputData，注入模拟数据）
3. Python ctypes 调用 DLL
4. 在游戏中测试

**参考**:
- https://github.com/nefarius/RawInput.Sharp
- https://docs.microsoft.com/en-us/windows/win32/inputdev/raw-input

**完成标准**: DLL 编译成功 + 游戏内鼠标响应

#### Task 0.4: 虚拟 HID 设备方案 [优先级 P2]
**预计**: 2-3 天  
**状态**: 未开始  
**前置**: Task 0.3 失败

**方案**: 用虚拟 USB 驱动模拟物理鼠标

**工具**:
- **ViGEmBus**: 虚拟游戏控制器驱动
- **Interception**: 键盘鼠标驱动级拦截
- **vJoy**: 虚拟游戏手柄

**优点**: 驱动级信号，游戏无法区分真假  
**缺点**: 安装复杂，可能被反作弊检测

**完成标准**: 安装驱动 + Python 能发送鼠标信号 + 游戏响应

#### Task 0.5: 视觉方案（备选）[优先级 P3]
**预计**: 1 天  
**状态**: 未开始  
**前置**: Task 0.3 和 0.4 均失败

**方案**: 不用鼠标事件，用 OpenCV 识别小地图方向

**执行步骤**:
1. 截取游戏画面，定位小地图区域
2. 识别人物朝向（箭头/三角形方向）
3. 计算目标方向 vs 当前方向的角度差
4. 用 SendInput 发送对应像素数的鼠标移动

**说明**: 视觉方案仍然需要某种鼠标输入方案，但可以降低精度要求（粗略转向即可）

---

### Phase 1: 输入验证 + 模型修正（输入解决后）

**目标**: 在输入可用的前提下，修正模型输出格式  
**时间**: 3-5 天  
**前置**: Phase 0 完成（至少一种输入方案可用）

#### Task 1.1: 过滤 idle 帧 + 重新训练 [P0]
**预计**: 3-4 小时  
**状态**: 脚本已就绪（filter_idle.py）

```powershell
# Step 1: 过滤 idle
C:\Python\python.exe filter_idle.py

# Step 2: 重新训练（离散量化方案）
C:\Python\python.exe -u training\goal_conditioned_bc_v53.py --data-dir data_noidle --epochs 50
```

**v5.3 模型设计**（基于用户决策：离散量化，不用连续值）:
```
动作空间（5类）:
  0: forward
  1: turn_left_slow   (dx ≈ -50px)
  2: turn_left_fast   (dx ≈ -150px)
  3: turn_right_slow  (dx ≈ +50px)
  4: turn_right_fast  (dx ≈ +150px)

损失: CrossEntropyLoss + Focal Loss
  class_weights = [1.0, 3.0, 3.0, 3.0, 3.0]  # 提升转向权重
```

**关键改进**:
- ✅ 转向有方向（left/right 分开）
- ✅ 转向有角度（slow/fast 两档）
- ✅ 过滤 idle（模型不学 idle）
- ✅ 离散量化（不回归，用分类）

**完成标准**: v5.3 训练完成 + 推理时能产生有效转向

#### Task 1.2: 验证输入+模型联动 [P0]
**预计**: 1-2 小时  
**前置**: Task 1.1 完成

**执行步骤**:
1. 用 Phase 0 找到的输入方案 + v5.3 模型
2. 在游戏中运行推理 60 秒
3. 记录：转向是否有方向？角度是否合理？导航是否流畅？

**完成标准**: 模型输出 → 鼠标移动 → 游戏响应，形成闭环

#### Task 1.3: DAgger 多轮迭代 [P1]
**预计**: 每天 1-2 小时，持续 3-5 天  
**前置**: Task 1.2 完成

**目标**: 干预率从 71.5% 降到 <25%

**执行步骤**:
1. 运行推理，人工纠正（F10）
2. 合并 DAgger 数据到训练集
3. 重新训练
4. 重复直到干预率 <25%

---

### Phase 2: 稳定 + 扩展（模型可用后）

**目标**: 导航模块稳定运行，开始战斗 AI  
**时间**: 2-4 周  
**前置**: Phase 1 完成

#### Task 2.1: 导航稳定性优化
- LSTM 时序建模（减少漂移）
- 辅助任务（深度预测、方向预测）
- 数据增强（随机裁剪、颜色抖动）

#### Task 2.2: 战斗 AI
- 识别敌人（YOLO / 模板匹配）
- 闪避时机（AutoDodge 已有框架）
- 攻击策略（轻击/重击/法术选择）

#### Task 2.3: PPO 强化学习
- 在 BC 预训练基础上做 RL
- 奖励函数：goal 距离 + 生存时间 + 伤害输出

---

## 📊 Phase 0 执行追踪

| Task | 状态 | 预计 | 实际 | 备注 |
|------|------|------|------|------|
| 0.1 研究 Turing 项目 | ⬜ | 2-3h | - | 第1优先 |
| 0.2 搜索其他项目 | ⬜ | 1-2h | - | 并行 |
| 0.3 Raw Input Hook | ⬜ | 1-2d | - | 前置: 0.1 |
| 0.4 虚拟 HID 设备 | ⬜ | 2-3d | - | 前置: 0.3 失败 |
| 0.5 视觉方案 | ⬜ | 1d | - | 前置: 0.3+0.4 失败 |

---

## 🚫 暂停的工作（等 Phase 0 完成）

- ❌ 模型架构优化（v5.3 暂不训练）
- ❌ 数据过滤（filter_idle.py 暂不运行）
- ❌ DAgger 多轮迭代
- ❌ LSTM / Transformer 引入
- ❌ PIDM 实现
- ❌ 传记系列第19-20本

**原因**: 同行评审指出，输入问题不解决，模型优化毫无意义。

---

## 📝 决策记录

| 日期 | 决策 | 原因 |
|------|------|------|
| 05-27 | 转向用离散量化，不用连续回归 | 用户决策：之前讨论过，量化更有利于训练 |
| 05-27 | 鼠标输入是 #1 优先级 | 同行评审 + 用户确认：阻塞整个项目 |
| 05-27 | 暂停模型/数据优化 | 同行评审建议：先攻坚后优化 |
| 05-26 | SendInput 部分有效 | 测试确认：Windows光标能动，但游戏用Raw Input |
| 05-25 | BC 失败根因诊断 | 时间对齐错误 + 数据分布失衡 + 协变量漂移 |

---

**最后更新**: 2026-05-27 07:44  
**当前阶段**: Phase 0 - 输入攻坚  
## Task 0.1 研究结论

### Turing 项目 (Black-Myth-Wukong-AI)
**结论：不解决鼠标问题**
- 战斗模块：纯键盘操作（J/M/O/K = 轻击/重击/跑/闪避），**不涉及鼠标**
- 跑图模块：用 GPT-4o 多模态 LLM 控制导航，不是视觉+鼠标方案
- 架构：识（捕捉画面）+ 算（预测出招）+ 触（交互）+ 探（跑图）+ 聚（数据）+ 斗战
- 对我们的价值：**零**，他们的方案完全绕过了鼠标控制问题

### PyDirectInput
**结论：与 SendInput 相同，已排除**
- 底层用的是 `SendInput()` win32 API
- 与我们已有的 mouse_util.py 方案相同
- 已在游戏中测试：无效（Raw Input 不读 SendInput）

### ViGEmBus 虚拟手柄方案 ⭐ 推荐
**结论：最有希望的方案**
- 原理：模拟 Xbox 360/DualShock 4 手柄，内核级 HID 设备
- 关键优势：黑神话**支持手柄**，右摇杆 = 视角控制，完全绕过鼠标 Raw Input
- 状态：项目已归档（2023.11 商标问题），但功能完全可用
- 最新版本：v1.22.0
- Python 库：`vigem`（需确认 Python 3.10 兼容性）

### Interception 驱动方案
**结论：备选方案**
- 原理：驱动级拦截+注入，可修改真实设备输入
- Python 绑定：`interception` 包 v0.6.0（仅支持 Python 3.13）
- 缺点：我们的环境是 Python 3.10，需升级或自行编译
- 更适合拦截/修改输入，不太适合凭空注入虚拟输入

### 方案优先级（更新）
1. ⭐ **ViGEmBus 手柄模拟** — 最简单、最可靠，游戏原生支持
2. **Interception 驱动** — 备选，需解决 Python 版本兼容
3. **虚拟 HID 鼠标驱动** — 门槛最高，需写 WDF 驱动
4. **DLL 注入 Hook Raw Input** — 复杂，需 C++ 开发

**最后更新**: 2026-05-27 07:54
**当前阶段**: Phase 0 - 输入攻坚  
**下一步**: Task 0.6 - 安装 ViGEmBus + 测试手柄模拟
