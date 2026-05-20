# PointNav模型快速上手

## 什么是PointNav模型

PointNav = **Point Goal Navigation**（点目标导航）

```
类比：
  PointNav: 给你一个GPS坐标(x,y)，让你走过去
  悟空AI: 给你一个存档点画面，让悟空走过去

本质相同：
  输入(当前状态 + 目标状态) → 输出(动作)
```

## 文件说明

```
models/pointnav_model.py    # 核心模型（DD-PPO编码器 + MiDaS深度 + Goal-Conditioned）
data_collection_pointnav.py # 数据采集器（采集当前帧+目标帧对）
```

## 快速开始

### Step 1: 安装依赖

```bash
# MiDaS（深度估计）
pip install timm

# 其他（应该已经装过了）
pip install torch torchvision opencv-python h5py numpy
```

### Step 2: 采集数据

```bash
# 方式1：命令行模式
python data_collection_pointnav.py --mode record --goal-dir savepoint_frames

# 方式2：Python脚本
from data_collection_pointnav import PointNavDataCollector

collector = PointNavDataCollector()

# 手动保存目标帧（存档点画面）
# 方法A：截取当前画面
frame = collector.screenshotter.capture()
collector.save_goal_frame(frame, name="savepoint_A")

# 方法B：指定图片文件
from PIL import Image
import numpy as np
img = Image.open("savepoint_screenshot.png")
collector.save_goal_frame(np.array(img), name="savepoint_A")

# 开始录制
collector.start_recording()

# ... 操作游戏 ...

# 保存轨迹
collector.stop_recording("trajectory_A_to_savepoint")
```

### Step 3: 训练

```bash
# 最简训练命令
python models/pointnav_model.py --train \
    --data-dir pathfinding_data \
    --model-path checkpoints/pointnav_model.pt \
    --epochs 50 \
    --use-depth

# 带参数的完整版本
python models/pointnav_model.py --train \
    --data-dir pathfinding_data \
    --goal-dir savepoint_frames \
    --model-path checkpoints/pointnav_model.pt \
    --latent-dim 256 \
    --hidden-dim 512 \
    --batch-size 8 \
    --lr 1e-3 \
    --epochs 100 \
    --use-depth \
    --device cuda
```

### Step 4: 推理

```python
from models.pointnav_model import PointNavInferer

# 加载模型
inferer = PointNavInferer(
    model_path="checkpoints/pointnav_model.pt",
    device="cuda"
)

# 设置目标（存档点画面）
inferer.set_goal("savepoint_frames/savepoint_A.png")

# 游戏循环中调用
while True:
    # 截取当前画面
    current_frame = screenshotter.capture()
    current_frame = preprocess(current_frame)  # (3, H, W) tensor
    
    # 获取动作
    result = inferer.step(current_frame)
    
    action_id = result["action_id"]
    mouse_delta = result["mouse_delta"]  # (dx, dy)
    direction = result["direction"]      # "forward" / "left" / "right" / "backward"
    
    # 执行动作
    execute_action(action_id)
    move_mouse(mouse_delta)
```

## 数据格式

### 训练数据（.h5文件）

```python
with h5py.File("pathfinding_data/trajectory_001.h5", "r") as f:
    frames = f["frames"][:]           # (T, 3, H, W) float32 [0, 1]
    actions = f["actions"][:]          # (T,) int64
    mouse_dx = f["mouse_dx"][:]        # (T,) float32
    mouse_dy = f["mouse_dy"][:]        # (T,) float32
    goal_frame = f["goal_frame"][:]    # (3, H, W) float32 [0, 1]
```

### 目标帧目录

```
savepoint_frames/
├── savepoint_A.png      # 存档点A的画面
├── savepoint_B.png      # 存档点B的画面
├── boss_entrance.png    # Boss房门口
└── ...
```

## 模型架构

```
输入：
  current_frame: (B, 3, 224, 224)  当前游戏画面
  goal_frame: (B, 3, 224, 224)    目标存档点画面
  depth: (B, 1, 224, 224)         深度图（可选）

编码器：
  DDPPO Visual Encoder (共享权重)
  ├── ResNet50 backbone (ImageNet预训练)
  ├── Depth conv (如果启用深度)
  └── 融合层 → 256维特征

融合：
  current_feat + goal_feat → 512维

输出：
  direction_logits: (B, 4)   方向预测
  action_logits: (B, num_actions)  动作分类
  mouse_pred: (B, 2)         鼠标移动
  distance_pred: (B, 1)      距离估计
```

## 配置参数

```python
# 模型
latent_dim = 256      # 特征维度
hidden_dim = 512      # 隐藏层维度
use_depth = True      # 是否使用深度估计

# 训练
lr = 1e-3             # 学习率
epochs = 50           # 训练轮数
batch_size = 8        # 批次大小

# 损失权重
weight_direction = 1.0   # 方向损失
weight_mouse = 2.0       # 鼠标损失
weight_distance = 0.5     # 距离损失
```

## 与旧版BC模型的区别

| 特性 | 旧版BC模型 | PointNav模型 |
|------|-----------|-------------|
| 目标 | 无 | 存档点画面 |
| 编码器 | ResNet18 (ImageNet) | ResNet50 (DD-PPO预训练) |
| 深度 | 无 | MiDaS深度估计 |
| 方向感知 | 无 | 方向预测头 |
| 适用任务 | 战斗/通用 | **导航专用** |
| 泛化能力 | 一般 | 更好（目标驱动） |

## 显存占用

```
RTX 2060 6GB 配置：

DD-PPO Encoder (ResNet50):
  参数: ~25M
  推理显存: ~300MB

MiDaS_small:
  参数: ~40M
  推理显存: ~160MB

PointNav模型 (总计):
  推理: ~500MB ✅
  训练 (batch=8): ~2GB ✅
```

## 常见问题

**Q: MiDaS加载失败？**
```python
# 方法1: 确保网络正常
pip install timm --upgrade

# 方法2: 手动下载权重
# https://github.com/intel-isl/MiDaS/releases
# 下载 MiDaS_small.pt 放到本地

# 方法3: 禁用深度
python models/pointnav_model.py --train --no-use-depth
```

**Q: 目标帧怎么获取？**
```python
# 方法1: 手动截图
# 玩游戏到存档点，按PrtSc，保存为savepoint_A.png

# 方法2: 自动提取
# 录制轨迹时，最后一帧自动作为目标

# 方法3: 用游戏内UI检测
# 检测到存档点UI时，自动截取
```

**Q: 如何处理第三人称视角？**
```python
# 模型已经内置处理：
# - direction_predictor: 预测目标在画面的哪个方向
# - mouse_predictor: 预测鼠标移动（控制视角）
# - 第三人称视角的"往前走"= "WASD移动 + 鼠标调整视角"
```

## 下一步

1. **采集数据**：至少10条存档点A→存档点B的轨迹
2. **训练模型**：50-100个epoch
3. **评估效果**：看模型是否能导航到目标
4. **迭代优化**：调整参数，采集更多数据

## 参考资料

- [Habitat PointNav](https://github.com/facebookresearch/habitat-lab)
- [DD-PPO Paper](https://arxiv.org/abs/1910.10838)
- [MiDaS Depth Estimation](https://github.com/intel-isl/MiDaS)
- [Vision-Language Navigation Survey](https://arxiv.org/abs/2204.09368)
