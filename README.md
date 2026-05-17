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

## 当前状态（2026-05-17）

### 训练进度
- **最佳模型**: `checkpoints/bc_best.pt`（43.5MB，epoch 23保存）
- **最佳准确率**: **74.02%**（epoch 23/50）
- **训练状态**: epoch 24时被SIGKILL中断（内存累积导致OOM）
- **鼠标输出**: ✅ 模型输出dx/dy，需真实游戏画面测试是否有意义

### 数据情况
- 12个h5文件，共8249训练样本
- 动作分布：idle 33.4% / forward 54.2% / right 7.5% / left 4.8% / dodge 0.0%
- 鼠标统计：dx mean=-0.006 std=0.244，dy mean=0.001 std=0.121

### 已知问题
1. 训练到~24 epoch因内存累积被kill（需在每个epoch后加`torch.cuda.empty_cache()`）
2. 数据forward偏重（54%），模型倾向一直往前走
3. 鼠标输出在随机噪声输入下几乎恒定，需真实游戏画面验证
4. `config.py`的`batch_size`临时改为16，训练完成后需改回64

### 下一步
1. 修复训练脚本内存泄漏 → 重新训练完整50 epochs
2. 推理测试（需要打开游戏）
3. 清理乱走数据或录更多转角数据平衡分布

---

## 四步流程（完整链路）

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

## License

MIT
