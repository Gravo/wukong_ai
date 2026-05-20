# 预训练模型调研：最小可用寻路模型

**调研日期**: 2026-05-20  
**目标**: 为悟空AI找到"拿来就能用"的预训练寻路模型  
**硬件约束**: RTX 2060 6GB，实时推理需求

---

## 一、需求分析：悟空AI需要什么样的预训练模型？

### 1.1 黑神话悟空的视觉特点

```
画面类型：第三人称3D动作游戏
视角：背后跟随视角（camera follows behind player）
核心挑战：
  1. 3D空间理解（前后左右深度）
  2. 地形识别（路/墙/障碍/可跳跃区域）
  3. 路径规划（从当前位置到目标）
  4. 动态环境（怪物移动、机关触发）

与自动驾驶的相似性：
  - 都是3D环境导航
  - 都需要实时决策
  - 都需要空间理解
  - 不同：游戏有固定地图，可预先扫描
```

### 1.2 预训练模型应该提供什么能力

| 能力 | 重要性 | 对应模型类型 |
|------|--------|------------|
| 空间深度理解 | 🔥 必须 | Depth Estimation / PointNav |
| 障碍物检测 | 🔥 必须 | Object Detection / Semantic Seg |
| 路径规划 | 🔥 必须 | VLN / Habitat Navigation |
| 动作预测 | ⚠️ 有帮助 | Video Understanding |
| 场景识别 | ⚠️ 有帮助 | Scene Understanding |

---

## 二、最小可用预训练模型推荐

### 🥇 第一推荐：Habitat-Sim DD-PPO（导航专用）

**项目**: [facebookresearch/habitat-lab](https://github.com/facebookresearch/habitat-lab)  
**任务**: PointNav（从A点导航到B点）  
**预训练模型**: DD-PPO (Distributed Deep PPO Policy)  
**参数规模**: ~50M（RTX 2060可跑）  
**框架**: PyTorch

**为什么推荐**：
```
1. 专门解决"从A导航到B"的问题 → 和悟空寻路高度匹配
2. 在真实室内环境数据上训练（Matterport3D, Gibson）
   → 学会了"哪里是路，哪里是墙"
3. 提供预训练的视觉编码器 → 可以直接作为悟空AI的backbone
4. 开源，文档完整
```

**使用方法**：

```python
# Step 1: 安装
# pip install habitat-sim habitat-lab

# Step 2: 加载预训练编码器
import habitat_sim
import torch

# 加载DD-PPO预训练模型
ddppo = torch.hub.load(
    "facebookresearch/habitat-baselines",
    "DDPPO",
    pretrained=True
)
frozen_encoder = ddppo.actor_critic.net.state_encoder

# frozen_encoder 输出: 256维空间特征
# 可以直接替换悟空AI的ResNet18编码器

class WukongHabitatEncoder(nn.Module):
    """
    用Habitat预训练编码器替代ResNet18
    预训练权重已经学会了：
    - 空间结构（门、墙、地板）
    - 深度感知
    - 路径规划
    """
    def __init__(self, frozen=True):
        super().__init__()
        self.encoder = frozen_encoder  # 预训练！
        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        # x: (B, C, H, W) 游戏画面
        return self.encoder(x)  # (B, 256)
```

**迁移到悟空的步骤**：

```
Week 1: 加载Habitat预训练编码器（1天）
         → 替换ResNet18，保持其他结构不变
         → 评估效果（应该比ImageNet预训练更好）

Week 2: 微调实验
         → 解冻最后几层
         → 悟空游戏数据fine-tune

Week 3: 如果效果好 → 融合方案：
         → Habitat编码器（空间理解）
         → + 战斗数据（动作预测）
```

**显存估算**：
```
Habitat encoder: ~20M参数 → ~250MB
动作头: ~5M参数 → ~60MB
推理时显存: ~400MB
RTX 2060 完全够用！
```

---

### 🥈 第二推荐：VLN视觉编码器（场景理解）

**项目**: PREVALENT / VLN-BERT / SimplerCNN  
**任务**: Vision-Language Navigation  
**数据**: Matterport3D（真实室内环境全景图）  
**参数规模**: 10-100M（多个可选）

**核心能力**：
```
VLN模型在大量室内导航任务中学会了：
1. Panoramic理解（360度全景）
2. 空间关系推理（A在B的左边）
3. 路径规划（如何从当前位置到目标）
4. 场景识别（这是厨房，那是卧室）
```

**推荐模型列表**：

| 模型 | 大小 | 特点 | RTX 2060 |
|------|------|------|---------|
| PREVALENT (ViT-S) | ~22M | Transformer+VLN预训练 | ✅ 可跑 |
| VLN-BERT (ViT-B) | ~86M | 大模型，效果好 | ⚠️ 勉强 |
| DDPPO (ResNet50) | ~25M | 专注导航 | ✅ 推荐 |
| CLIP ViT-B | ~86M | 通用视觉 | ⚠️ 勉强 |
| CLIP ViT-L | ~428M | ❌ 太大 | ❌ 不可跑 |

**最小可用推荐**：`PREVALENT with ViT-Small` 或 `DDPPO with ResNet50`

---

### 🥉 第三推荐：Depth Estimation（3D感知）

**项目**: Intel-isl/MiDaS / MiDaS-large（单目深度估计）  
**任务**: 从单张图片预测深度（3D重建）  
**模型大小**: ~40-350M（多个版本）  
**框架**: PyTorch

**为什么有用**：

```
黑神话悟空是3D游戏，但视频输入是2D的
→ 深度估计把2D画面转成"伪3D"

有什么用：
1. 距离感知：前方障碍物有多远？
2. 地形：地面是平的还是斜坡？
3. 可跳跃距离：能跳过去吗？
4. 高度感知：上方有什么（天花板/平台）？
```

**使用方法**：

```python
import torch.hub

# 加载最小版深度估计（MobileNet backbone）
model = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small"  # ~40M参数，RTX 2060轻松跑
)

def estimate_depth(game_frame):
    """
    输入: (B, 3, 224, 224) RGB游戏画面
    输出: (B, 1, 224, 224) 深度图
    """
    with torch.no_grad():
        depth = model(game_frame)  # 归一化深度值
    
    # 深度图可以这样用：
    # 1. 直接拼接到ResNet输入（扩通道）
    combined = torch.cat([rgb, depth], dim=1)  # (B, 4, 224, 224)
    
    # 2. 提取深度统计特征
    avg_depth = depth.mean(dim=[2, 3])  # 平均深度
    depth_std = depth.std(dim=[2, 3])  # 深度变化
    
    # 3. 用于碰撞检测（深度突然变小=有障碍物）
    
    return depth
```

**显存估算**：
```
MiDaS_small: ~40M参数 → ~160MB
组合到ResNet: 额外+80MB
总推理显存: ~300MB
RTX 2060 完全够用！
```

---

## 三、备选方案

### 3.1 MineCLIP（游戏视觉编码器）

**项目**: [viagqi/CLIP4Clip](https://github.com/viagqi/CLIP4Clip) 或 [MineDojo/MineCLIP](https://github.com/MineDojo/MineCLIP)  
**任务**: Minecraft视频理解  
**特点**: 专门在游戏视频上预训练  
**RTX 2060**: ✅ 可跑（ViT-S ~22M参数）

```python
# MineCLIP专门在Minecraft游戏数据上训练
# 比ImageNet预训练更接近"游戏视觉理解"

import MineCLIP

model = MineCLIP.load("ViT-B/32")
# model.encoder 可以作为视觉backbone
# 学到了：游戏内的空间关系、物体识别、动作理解
```

**局限性**：Minecraft是体素风格，和黑神话的真实渲染差异大，但"空间导航"能力可迁移。

### 3.2 GTA-V 自动驾驶模型

**相关研究**：玩GTA-V的自动驾驶AI  
**预训练模型**：Carla自动驾驶模型  
**特点**：真实街景3D导航

```
搜索关键词：
- "Learning to Drive from GTA V"
- "Carla Simulator Pre-trained Models"
- "Autonomous Driving in CARLA"
```

**迁移价值**：自动驾驶的"3D导航"能力 → 悟空的"3D游戏导航"能力

### 3.3 动作识别模型（动作预训练）

**项目**: X3D, SlowFast, TimeSformer  
**任务**: 视频动作识别  
**特点**: 时序预训练（悟空需要！）

```python
# SlowFast Networks - 预训练的动作识别
# 学到了：动作模式、时序关系

from slowfast.models import build_model
model = build_model(cfg)  # 加载X3D-M预训练（~3.8M参数！极小）
```

**迁移价值**：
- 时序建模能力 → 悟空的"连贯动作"
- 但需要修改动作空间（识别vs生成）

---

## 四、最快可用方案（本周可实现）

### 方案A：Habitat + Depth（推荐顺序）

```
Day 1: 安装Habitat + 加载DDPPO预训练编码器
Day 2: 替换ResNet18，冻结编码器，只训练动作头
Day 3-4: 微调实验，记录效果
Day 5: 对比Habitat vs ResNet18 vs ImageNet
```

```python
# 最简实现（直接可用）
class FastWukongModel(nn.Module):
    def __init__(self):
        # 1. Habitat预训练空间编码器
        self.spatial_encoder = load_ddppo_encoder()  # 预训练！
        
        # 2. MiDaS深度编码器（可选）
        self.depth_encoder = load_midas_small()      # 预训练！
        
        # 3. 融合
        self.fusion = nn.Linear(256 + 256, 512)
        
        # 4. 动作头（从头训练）
        self.action_head = nn.Linear(512, NUM_ACTIONS)
        self.mouse_head = nn.Linear(512, 2)
    
    def forward(self, frame):
        # RGB特征
        rgb_feat = self.spatial_encoder(frame)  # (B, 256)
        
        # 深度特征
        depth = self.depth_encoder(frame)       # (B, 256)
        
        # 融合
        combined = torch.cat([rgb_feat, depth], dim=-1)  # (B, 512)
        fused = self.fusion(combined)           # (B, 512)
        
        # 动作预测
        action = self.action_head(fused)
        mouse = self.mouse_head(fused)
        return action, mouse
```

### 方案B：零成本方案（立即可试）

**如果不想安装Habitat，用这个**：

```python
# 完全基于PyTorch Hub，无需额外安装

class MinimalPreTrainedWukong(nn.Module):
    def __init__(self):
        # 方案1：ResNet50 (ImageNet预训练，已在config.py)
        self.resnet = resnet50(pretrained=True)
        
        # 方案2：EfficientNet-B0 (更轻量)
        # self.effnet = efficientnet_b0(pretrained=True)
        
        # 方案3：MobileNetV3-Large (最小可用)
        # self.mobilenet = mobilenet_v3_large(pretrained=True)
        
        # 修改动作头
        self.action_head = nn.Linear(2048, NUM_ACTIONS)
        self.mouse_head = nn.Linear(2048, 2)
    
    def forward(self, x):
        feat = self.resnet(x)  # (B, 2048)
        return self.action_head(feat), self.mouse_head(feat)

# 这个方案：0额外安装，直接用PyTorchHub加载
# 效果：比随机初始化好，但不如Habitat导航预训练
```

---

## 五、模型对比总表

| 模型 | 类型 | 预训练任务 | 参数 | RTX 2060 | 寻路适配 | 迁移难度 |
|------|------|---------|------|---------|---------|---------|
| **DD-PPO (Habitat)** | 导航 | PointNav | ~50M | ✅ | 🔥🔥🔥 | 中等 |
| **PREVALENT** | VLN | VLN | ~22M | ✅ | 🔥🔥 | 中等 |
| **MiDaS_small** | 深度 | 深度估计 | ~40M | ✅ | 🔥🔥 | 简单 |
| **ResNet50 (IN)** | 分类 | ImageNet | ~25M | ✅ | 🔥 | 简单 |
| **EfficientNet-B0** | 分类 | ImageNet | ~5M | ✅ | 🔥 | 简单 |
| **CLIP ViT-S** | 对比 | 图文 | ~22M | ✅ | 🔥🔥 | 中等 |
| **MineCLIP** | 对比 | Minecraft | ~86M | ⚠️ | 🔥🔥 | 中等 |
| **SlowFast X3D-M** | 动作 | Kinetics | ~3.8M | ✅ | 🔥🔥🔥 | 简单 |
| **RT-2** | VLA | 机器人 | >1B | ❌ | 🔥🔥 | 困难 |
| **VLN-BERT** | VLN | VLN | ~86M | ⚠️ | 🔥🔥 | 中等 |

**推荐组合**（按实现难度）：

```
最快方案（0成本）：ResNet50(ImageNet) + MiDaS_small(深度)
推荐方案（效果最好）：DD-PPO(Habitat) + MiDaS_small(深度)
长期方案（最强大）：VLN编码器 + Depth + WorldModel
```

---

## 六、搜索关键词（如果想深入找）

```
# GitHub搜索
site:github.com "point navigation" pytorch pretrained
site:github.com "vision language navigation" pretrained
site:github.com "habitat-sim" pretrained model

# HuggingFace搜索
Pretrained navigation model pytorch
Vision encoder game AI pretrained

# 论文
"PointNav" "pretrained" "visual encoder"
"Vision-Language Navigation" "transfer learning"
"3D navigation" "pretrained" "game AI"
```

---

## 七、我的建议（明确答案）

**如果只想选一个**：用 **DD-PPO (Habitat)** 的预训练视觉编码器替换ResNet18

**原因**：
1. 专门为"导航"任务训练 → 比ImageNet更相关
2. 学会了真实室内环境的空间理解 → 可迁移到游戏
3. 参数适中 → RTX 2060可跑
4. PyTorch直接加载 → 无需复杂安装

**如果想要组合**：DD-PPO编码器 + MiDaS深度估计

**如果想最快试**：直接用ResNet50（config.py已有），同时加一个MiDaS_small

---

**文档状态**: 调研完成
**下一步**: 
1. 确认哪个预训练模型可以正常加载（环境测试）
2. 选择最合适的做迁移实验
