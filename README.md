# 🐒 wukong_ai

Black Myth: Wukong (黑神话：悟空) AI - 行为克隆寻路 + PPO战斗

当前目标：**虎先锋 (Tiger Vanguard)**

## 架构

```
wukong_ai/
├── config.py                    # 集中配置（超参数、窗口坐标、动作空间）
├── requirements.txt
├── env/
│   ├── screen_capture.py       # 高速截图 (dxcam/mss/win32)
│   ├── blood_detector.py       # HSV色域血量检测
│   └── action_executor.py      # pydirectinput动作执行
├── models/
│   ├── resnet_encoder.py       # ResNet18视觉编码器
│   └── ppo_agent.py            # PPO Actor-Critic
├── training/
│   ├── train_combat.py         # 战斗训练主脚本
│   └── data_collector.py       # 人类Demo数据采集 v2.1
├── pathfinding/
│   ├── preprocess_data.py      # h5 → npz预处理
│   ├── behavior_clone_v2.py    # 行为克隆v2（鼠标是输出）
│   └── inference_v2.py         # 推理v2（实时控制角色）
├── checkpoints/                # 模型保存目录
│   └── bc_best.pt              # 行为克隆模型
└── pathfinding_data/           # 录制数据目录
    ├── *.h5                    # 原始录制数据
    └── preprocessed/
        └── stacked_data.npz    # 预处理后的训练数据
```

## 当前状态（2026-05-19）

### 问题诊断

**核心问题**：模型收敛但未学到有效动作
- 行为克隆准确率达到 74%，但推理时只会往前走
- 鼠标输出几乎恒定，视角控制失效
- 模型选择了"最安全"的策略：idle 或 forward

**根因**：
1. 数据分布失衡：idle 33.4% + forward 54.2% = 87.6%
2. Distribution Shift：BC 只学 state→action，不理解后果
3. 鼠标稀疏性：大部分帧鼠标不动，MSE 损失被淹没
4. 损失函数不均衡：分类损失 vs 回归损失量级不同

### 改进方案

**已完成**：
- ✅ 创建 `behavior_clone_v3.py`：数据过滤 + LR 调度 + 鼠标损失加权
- ✅ 提取共享模型定义到 `models/bc_model.py`
- ✅ 修复内存泄漏问题（gc.collect + cuda.empty_cache）
- ✅ 添加详细分析文档 `docs/ANALYSIS.md`

**待执行**：
- 在有游戏环境的机器上运行 v3 训练
- 推理测试验证效果
- 如果效果不够，实施 DAgger 在线学习

### 训练进度
- **v2 模型**: `checkpoints/bc_best.pt`（74.02% 准确率，但效果不佳）
- **v3 模型**: 待训练（使用数据过滤 + LR 调度）

---

## 快速开始（v3 改进版）

### 1. 分析数据分布
```bash
python pathfinding/behavior_clone_v3.py --analyze-only
```

### 2. 训练改进模型
```bash
# 默认配置（idle 保留 10%，鼠标权重 2.0x）
python pathfinding/behavior_clone_v3.py

# 自定义配置
python pathfinding/behavior_clone_v3.py --epochs 100 --idle-ratio 0.1 --mouse-weight 2.0
```

### 3. 推理测试
```bash
# 打开游戏，进入虎先锋战斗
python pathfinding/inference_v2.py --duration 120 --fps 10
```

---

## 完整流程（四步链路）

### 1. 采集数据

录制人类玩家的游戏画面+操作（WASD、J攻击、Space闪避、鼠标转视角等）：

```powershell
cd D:\projects\wukong_ai

# 基本录制（5分钟，15fps，ESC停止或到时间自动停）
C:\Python\python.exe -u training/data_collector.py --duration 300 --fps 15

# 跳过输入测试（已确认环境OK时）
C:\Python\python.exe -u training/data_collector.py --duration 300 --fps 15 --skip-test

# 带HUD预览（会弹出cv2窗口）
C:\Python\python.exe -u training/data_collector.py --duration 300 --fps 15 --hud
```

**参数说明**：
- `--duration 300`：每个episode最长300秒（5分钟），到时间自动停止
- `--fps 15`：录制帧率（15fps足够，减少数据量）
- `--auto-save 3000`：每3000帧自动保存一次（防止kill丢数据）
- `--skip-test`：跳过输入设备测试（已确认keyboard+mouse就绪时）
- `--hud`：启用HUD预览窗口（默认关闭）

**停止方式**：
- 按 **ESC** 停止（keyboard全局hook，游戏内可靠）
- 或等到 `--duration` 时间到自动停止

**录制要求**：
- 至少录制 **5分钟**
- 覆盖 **多种动作**（WASD移动、J攻击、Space闪避、鼠标转视角）
- idle占比 **<70%**
- 鼠标活跃度 **>5%**

**注意**：keyboard库需要**管理员权限**才能全局hook。如果ESC不生效，请用管理员终端运行。

### 2. 预处理数据

将h5原始数据转换为预堆叠npz格式：

```powershell
C:\Python\python.exe pathfinding/preprocess_data.py
```

输出：`pathfinding_data/preprocessed/stacked_data.npz`

### 3. 训练模型

行为克隆训练（ResNet18编码器 → 分类头 + 回归头）：

```powershell
C:\Python\python.exe pathfinding/behavior_clone_v2.py
```

输出：`checkpoints/bc_best.pt`

### 4. 推理测试

实时控制角色：

```powershell
C:\Python\python.exe pathfinding/inference_v2.py --duration 60 --fps 10
```

推理时会自动：
- 检测游戏窗口位置
- 激活游戏窗口
- 根据画面输出动作+鼠标移动

---

## 质量报告

采集结束后会自动输出质量报告：

```
============================================================
  录制质量报告
============================================================

  Episode 1: 4500 frames, 300.0s, FPS=15.0
    动作分布:
      idle: 1584 (35.2%)
      forward: 1200 (26.7%)
      attack: 900 (20.0%)
      right: 516 (11.5%)
      dodge: 300 (6.7%)
    鼠标活跃: 1275/4500 (28.3%)

  ─────────────────────────────────────────
  总计: 4500 帧, 300.0s (5.0min)

  动作分布（汇总）:
    idle         1584 ( 35.2%) ██████████████░░░░░░░░░░░░░░░░░░░░
    forward      1200 ( 26.7%) ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░
    attack        900 ( 20.0%) ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    right         516 ( 11.5%) ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    dodge         300 (  6.7%) ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

  鼠标活跃帧: 1275/4500 (28.3%)

  合格判定:
    ✅ 数据量: 4500帧
    ✅ idle占比: 35.2%
    ✅ 动作多样性: 5种非idle动作
    ✅ 鼠标活跃度: 28.3%

  🎉 录制质量合格！可以进行预处理和训练。

  下一步:
    1. 预处理: C:\Python\python.exe pathfinding/preprocess_data.py
    2. 训练:   C:\Python\python.exe pathfinding/behavior_clone_v2.py
    3. 推理:   C:\Python\python.exe pathfinding/inference_v2.py --duration 60 --fps 10
============================================================
```

**合格标准**：
- 数据量 ≥ 3000帧
- idle占比 < 70%
- 非idle动作 ≥ 3种
- 鼠标活跃度 > 5%

---

## 技术亮点

| 项目 | 实现 |
|------|------|
| 截图 | dxcam (120fps+) / mss / win32 |
| 动作执行 | pydirectinput (DirectX兼容) |
| 视觉编码器 | ResNet18 (ImageNet预训练) |
| 行为克隆 | 分类头(动作) + 回归头(鼠标dx/dy) |
| 血量检测 | HSV色域分割 |
| RL算法 | PPO (Actor-Critic) |

---

## 动作空间

| ID | 动作 | 按键 |
|----|------|------|
| 0 | Idle | - |
| 1 | Attack | J |
| 2 | Heavy Attack | J×4 |
| 3 | Dodge | Space |
| 4 | Move Forward | W |
| 5 | Move Right | D |
| 6 | Move Left | A |
| 7 | Dodge + Attack | Space + J |
| 8 | Lock On | V |
| 9 | Heal (Gourd) | R |

---

## 详细分析

关于当前方案的问题诊断、替代方案研究（OpenVLA、DAgger、GAIL、Diffusion Policy 等）、以及推荐实施路径，请参考：

**[docs/ANALYSIS.md](docs/ANALYSIS.md)**

---

## License

MIT
