# 🐒 wukong_ai

Black Myth: Wukong (黑神话：悟空) — Goal-Conditioned Behavior Cloning + ViGEmBus Gamepad Control

**当前目标**：端到端自主寻路（Point Navigation）

---

## 架构总览

```
wukong_ai/
├── config.py                         # 集中配置
├── requirements.txt
│
├── env/
│   ├── screen_capture.py             # dxcam / mss / win32 截图
│   ├── blood_detector.py            # HSV 血量检测
│   └── action_executor.py           # pydirectinput 动作执行
│
├── training/
│   ├── goal_conditioned_bc_v55_optimized.py   # ✅ v5.5 训练脚本（当前最佳）
│   ├── inference_goal_v56.py                 # ✅ v5.6 推理（ViGEmBus 手柄版）
│   ├── data_collector_v3.py                  # 数据采集器 v3
│   ├── filter_idle.py                         # 过滤 idle 帧
│   ├── analyze_data.py                        # 训练数据分析
│   ├── test_bucket_mapping.py                 # 手柄映射测试
│   └── checkpoints/
│       └── goal_bc_v55_best_acc_a.pt        # ✅ 推荐模型（Val Acc_A=94.09%）
│
├── pathfinding_data/                # 原始录制数据（h5）
├── pathfinding_data_balanced/       # 平衡后数据（oversample + Focal Loss）
│
├── docs/                           # 技术分析文档
│   ├── TECHNICAL_ANALYSIS.md
│   ├── RESEARCH_BC_FAILURE_ANALYSIS.md
│   ├── VLA_Research.md
│   └── ...
│
└── checkpoints/                    # 所有模型 checkpoint
```

---

## 当前状态（2026-06-07）

### ✅ 已完成

- **v5.5 训练完成**（2026-05-28）
  - 数据：9,790 帧（oversample 后），19 个 h5 文件
  - 架构：Goal-Conditioned BC，双头（Action Head 3类 + Mouse Head 7-bucket）
  - 指标：Train Acc 99.94% / Val Acc_A 94.09% / Val Acc_M 93.15%
  - 过拟合可控（~5% 差距）
  - 模型：`checkpoints/goal_bc_v55_best_acc_a.pt`（推荐）

- **ViGEmBus 手柄方案验证通过**（2026-06-07）
  - 游戏使用 Raw Input API → pydirectinput / SendInput 无效
  - 解决方案：ViGEmBus + `vgamepad` 库，模拟 Xbox 360 手柄
  - 右摇杆控制视角，左摇杆控制移动
  - 测试脚本：`training/test_bucket_mapping.py`

### ⚠️ 已知问题

1. **goal_id 全部为 0**（录制时未按 G 键切换 goal）
   - 影响：模型实际是普通 BC，非 Goal-Conditioned
   - 修复：重新录制时按 G 键切换 goal_id

2. **双头预测冲突**（Action Head vs Mouse Head）
   - 修复：推理时统一用 Mouse Head（7-bucket）决定 action
   - 已在 `inference_goal_v56.py` 中实现

3. **bucket 映射需实测校准**
   - `BUCKET_STICK = {0:-0.80, 1:-0.50, ..., 6:+0.80}`
   - 如果方向反转，修改符号

---

## 快速开始

### 1. 环境准备

```powershell
# Python 环境（项目使用 C:\Python\python.exe，3.10.10）
C:\Python\python.exe -m pip install torch==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
C:\Python\python.exe -m pip install dxcam mss opencv-python pyautogui keyboard vgamepad
```

**vgamepad 手动安装**（PyPI 无包）：
1. 下载 `vgamepad-0.0.6.tar.gz`
2. 解压到 `C:\Python\Lib\site-packages\vgamepad\`
3. 确认 `win\vigem\client\x64\ViGEmClient.dll` 存在

**ViGEmBus 驱动**：安装 [Nefarius ViGEmBus](https://github.com/ViGEm/ViGEmBus)

---

### 2. 测试手柄映射

```powershell
cd D:\projects\wukong_ai\training
C:\Python\python.exe test_bucket_mapping.py
```

聚焦游戏窗口，观察：
- bucket 0-2 → 视角左转
- bucket 3   → 视角不动
- bucket 4-6 → 视角右转

如果方向反转，修改 `test_bucket_mapping.py` 和 `inference_goal_v56.py` 中的 `BUCKET_STICK` 符号。

---

### 3. 推理运行

```powershell
cd D:\projects\wukong_ai\training
C:\Python\python.exe inference_goal_v56.py --duration 120
```

**参数**：
- `--model`：模型路径（默认 `checkpoints/goal_bc_v55_best_acc_a.pt`）
- `--goal-id`：目标 ID（默认 0，当前数据全部为 0）
- `--duration`：运行时长（秒）
- `--conf-threshold`：置信度阈值（默认 0.5）
- `--device`：设备（默认 `cuda:0`）

**推理时**：
- 左摇杆 Y=1.0 持续前进
- 右摇杆 X 轴按 bucket 映射转动视角
- EMA 平滑系数 0.7（避免抖动）

---

## 训练流程

### 1. 采集数据

```powershell
cd D:\projects\wukong_ai\training
C:\Python\python.exe data_collector_v3.py --duration 300 --fps 15
```

**注意**：
- 按 **G 键**切换 goal_id（会在文件名中记录）
- 按 **ESC** 停止录制（keyboard 全局 hook）
- 需要**管理员权限**运行（keyboard 库要求）

### 2. 过滤数据

```powershell
C:\Python\python.exe filter_idle.py
```

过滤掉 `action==0`（idle）的帧，输出到 `pathfinding_data_noidle/`。

### 3. 训练模型

```powershell
cd D:\projects\wukong_ai\training
C:\Python\python.exe goal_conditioned_bc_v55_optimized.py
```

**输出**：
- `checkpoints/goal_bc_v55_best_loss.pt`
- `checkpoints/goal_bc_v55_best_acc_a.pt`（推荐）
- `checkpoints/goal_bc_v55_best_acc_m.pt`

---

## 技术分析文档

| 文档 | 内容 |
|------|------|
| `docs/TECHNICAL_ANALYSIS.md` | 技术架构分析 |
| `docs/RESEARCH_BC_FAILURE_ANALYSIS.md` | BC 失败根因分析 |
| `docs/VLA_Research.md` | VLA 架构调研 |
| `docs/PATHFINDING_PROBLEM.md` | 寻路问题定义 |
| `training/TRAINING_REPORT.md` | 训练报告 |
| `PROJECT_SUMMARY_v55.md` | v5.5 项目总结 |

---

## 关键决策记录

### 为什么不用纯 RL？
- 样本效率低，训练时间长
- 黑神话高维视觉输入 + 稀疏奖励，难以收敛

### 为什么用 DAgger？
- 比纯 BC 样本效率高
- "模型操作 + 人类纠正"覆盖错误状态分布
- 但干预率 71.5% 偏高，需迭代至 <25%

### 为什么用 ViGEmBus？
- 游戏使用 Raw Input API 读取鼠标
- pydirectinput / SendInput 只影响 Windows 光标，游戏内无效
- ViGEmBus 模拟 Xbox 360 手柄，绕过 Raw Input 过滤

### 为什么用双头模型？
- Action Head（3类）：forward / turn_left / turn_right
- Mouse Head（7-bucket）：更精细的转向控制
- 推理时统一用 Mouse Head 决定 action，避免双头冲突

---

## 硬件要求

| 组件 | 最低要求 | 推荐 |
|------|----------|------|
| GPU | RTX 2060 6GB | RTX 3060 12GB+ |
| RAM | 16GB | 32GB |
| 存储 | 10GB | 50GB SSD |

**注意**：训练时系统内存占用 ~4.5GB（预加载数据到内存），GPU 显存占用 ~2GB。

---

## 许可证

MIT

---

## 贡献

欢迎提交 Issue / PR！

**待改进**：
- [ ] 重新录制带 goal_id 的数据
- [ ] 实测校准 bucket → 右摇杆映射
- [ ] DAgger 迭代降低干预率
- [ ] 引入 LSTM 处理时序依赖
- [ ] 奖励函数设计（RL 方案）

---

**项目仓库**：https://github.com/Gravo/wukong_ai