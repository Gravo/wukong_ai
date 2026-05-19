# wukong_ai 深度研究：自动驾驶架构演进 + 语音LLM时序建模

**研究日期**: 2026-05-19

---

## 一、自动驾驶架构演进与启示

> 用户洞察（豆包）：当前 wukong_ai 的架构类似第一代自动驾驶，主要问题是训练无法收敛，模型没学到图片语义。

这个判断非常准确。自动驾驶领域用了 10 年走完的路，wukong_ai 正在重复同样的错误。

### 1.1 第一代：端到端（2015-2017）← wukong_ai 当前位置

**代表工作**：
- **NVIDIA PilotNet (2016)**：CNN，输入前摄像头图像，输出转向角度
- **comma.ai (2016)**：端到端驾驶辅助

**方法**：
```
摄像头图像 → CNN → 转向角/加速度
```

**核心问题**（和 wukong_ai 完全一样）：
| 问题 | 自动驾驶表现 | wukong_ai 表现 |
|------|--------------|----------------|
| 不可解释 | 不知道模型为什么左转 | 不知道模型为什么一直 forward |
| 不泛化 | 换一条路就不会开 | 换一个场景就不会走 |
| Compounding Error | 小偏差累积，越走越偏 | 同上 |
| 学到作弊特征 | 看到车道线就转，不是理解道路结构 | 看到某颜色/纹理就 forward，不是理解可通行性 |

**结论**：纯端到端 BC 在第一代自动驾驶已经被证明不行。wukong_ai 正在踩同样的坑。

### 1.2 第二代：模块化（2018-2022）

**架构**：
```
感知模块（Perception）
  ↓ 检测：车道线、障碍物、交通标志
规划模块（Planning）
  ↓ 基于感知结果：A*、RRT、学习式规划
控制模块（Control）
  ↓ 执行轨迹：PID、MPC
```

**代表系统**：
- **Apollo（百度）**：感知（CNN）→ 规划（动态规划）→ 控制
- **Autoware**：开源自动驾驶框架
- **Wayve（早期版本）**：模块化端到端

**优点**：
- ✅ 可解释（能看到每个模块的输出）
- ✅ 各模块可独立改进
- ✅ 感知可以用经典 CV + 深度学习

**缺点**：
- ❌ 模块间误差传播（感知错了，规划一定错）
- ❌ 感知层手工设计特征

### 1.3 第三代：端到端 + 学习式中间表示（2023-）

**代表**：
- **Tesla FSD v12**：据称是单一端到端神经网络，但有中间表示
- **Wayve（最新）**：端到端学习，但有可解释的中间表示

**关键洞察**：
```
第一代：黑盒（不可解释）
第二代：手工中间表示（可解释但次优）
第三代：学习的的中间表示（可解释 + 最优）
```

中间表示是**让模型自己学**，不是手工设计：
```
输入图像 → [中间表示：车道线/障碍物/可行驶区域] → 控制信号
            ↑ 这些是模型自己学的，不是手工标注的
```

**对 wukong_ai 的意义**：
- 当前方案（纯 BC）= 第一代，已经证明不行
- 应该迈向：端到端 + **学习的**中间表示
- Goal 变量就是一种学习的中间表示

### 1.4 对 wukong_ai 的具体启示

| 自动驾驶演进 | wukong_ai 对应改进 |
|-------------|-------------------|
| 第一代（端到端）失败 | ✅ 已确认：当前 BC 失败 |
| 添加中间表示（车道线、障碍物） | 🔲 添加 goal 变量（目标位置） |
| 感知解耦（单独训练感知模块） | 🔲 MoE：战斗专家 + 寻路专家 |
| 时序建模（光流、轨迹预测） | 🔲 添加 LSTM/Transformer 时序头 |

**结论**：wukong_ai 不应该在纯 BC 上修修补补，而应该参考自动驾驶第三代架构：**端到端 + 学习的的中间表示**。

---

## 二、李沐的语音大语言模型研究

### 2.1 李沐是谁

- 亚马逊首席科学家（Principal Scientist）
- 《动手学深度学习》（Dive into Deep Learning / d2l）作者之一
- B站UP主（账号：跟李沐学AI），有大量深度学习教学视频
- 研究方向：深度学习系统、语音处理、大语言模型

### 2.2 关键论文：《Back to Basics: Revisiting ASR in the Age of Voice Agents》

- **发表时间**：2026-03-26（arXiv）
- **作者**：Geeyang Tay, Wentao Ma, Jaewon Lee, Yuzhi Tang, Daniel Lee
- **核心内容**：系统研究语音识别（ASR）在语音 Agent 时代的鲁棒性问题

**关键发现**：
- 精选基准测试上 ASR 准确率接近人类，但真实场景（环境噪声、口音、方言）下严重退化
- 提出 WildASR 基准：从真实人类语音采集，涵盖环境变量、人口统计偏移、语言多样性三个维度
- 评估了 7 个广泛使用的 ASR 系统，发现严重且不均匀的性能退化

**对 wukong_ai 的启示**：
> 在精选数据集上训练，不等于在真实场景可用。wukong_ai 的数据采集应该覆盖多样化场景（不同光照、不同敌人、不同地形），而不是只在一种情况下采集。

### 2.3 语音LLM的时序建模（核心：为什么用户说「会有很大帮助」）

语音的本质是**时间序列**。语音LLM必须解决的核心问题和 wukong_ai 完全一样：

```
语音LLM：    语音信号（时间序列）→ 理解 → 生成回复
wukong_ai：  游戏画面（时间序列）→ 理解 → 生成动作
```

**语音LLM的典型时序建模方法**：

| 方法 | 代表模型 | 特点 | 对 wukong_ai 的参考价值 |
|------|---------|------|------------------------|
| RNN（LSTM/GRU） | 早期语音识别 | 简单，适合短序列 | 可以尝试在 ResNet 后加 LSTM 层 |
| Transformer（Attention） | Whisper、Qwen-Audio | 长程依赖，强大 | 用 Transformer 替代 CNN 部分层 |
| 状态空间模型（Mamba） | Mamba、S4 | 高效，适合超长序列 | 如果帧数很多（>100帧），Mamba 比 Transformer 高效 |
| 向量量化（VQ-VAE） | SoundStream、EnCodec | 离散化连续信号 | 可以把画面离散化成「视觉词」，再用 LLM 处理 |

### 2.4 Qwen2-Audio 架构分析（李沐在B站讲解过）

从 GitHub 获取的 Qwen2-Audio 信息：

**模型结构**（三阶段训练）：
```
阶段1: 音频编码器预训练
  音频波形 → 音频编码器（自监督） → 音频表示

阶段2: 音频-语言对齐
  音频表示 + 文本 → 跨模态对齐 → 多模态模型

阶段3: 指令微调
  音频 + 文本指令 → Qwen2 LLM → 文本回复
```

**两种交互模式**：
1. **Voice Chat**：用户直接语音输入，无需文本
2. **Audio Analysis**：用户提供音频 + 文本指令，模型分析音频

**关键时序建模技术**：
- 音频编码器：处理时序音频信号（通常 16kHz，每秒 16000 个采样点）
- 跨模态对齐：将时序音频特征对齐到离散文本 token
- LLM：Transformer 注意力机制，建模长程时序依赖

### 2.5 对 wukong_ai 的具体改进建议

当前 wukong_ai 的架构：
```
4帧堆叠 → ResNet18 → 256维特征 → 动作分类头
                      ↓
                  鼠标回归头
```

**问题**：ResNet18 是纯空间特征提取器，完全没有时序建模能力。4帧堆叠只是简单拼接，不是真正的时序建模。

**改进方案A（简单，推荐先试）**：在 ResNet 后加 LSTM：
```python
class TemporalBC(nn.Module):
    def __init__(self):
        self.encoder = ResNet18(pretrained=True)  # 输出 256维
        self.lstm = nn.LSTM(256, 512, num_layers=2, batch_first=True)
        self.action_head = nn.Linear(512, NUM_ACTIONS)
        self.mouse_head = nn.Linear(512, 2)

    def forward(self, frame_seq):
        # frame_seq: [B, T, C, H, W]
        B, T = frame_seq.shape[:2]
        feats = []
        for t in range(T):
            f = self.encoder(frame_seq[:, t])  # [B, 256]
            feats.append(f)
        feats = torch.stack(feats, dim=1)  # [B, T, 256]
        _, (h_n, _) = self.lstm(feats)      # h_n: [2, B, 512]
        feat = h_n[-1]                        # [B, 512] 取最后一层
        return self.action_head(feat), self.mouse_head(feat)
```

**改进方案B（更强但需要更多数据）**：用 Transformer 替代 LSTM：
```python
class TransformerBC(nn.Module):
    def __init__(self):
        self.encoder = ResNet18(pretrained=True)
        self.pos_embed = nn.Parameter(torch.randn(1, T_max, 256))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=4
        )
        self.action_head = nn.Linear(256, NUM_ACTIONS)

    def forward(self, frame_seq):
        B, T = frame_seq.shape[:2]
        feats = self.encoder(frame_seq.view(B*T, ...)).view(B, T, 256)
        feats = feats + self.pos_embed[:, :T]
        feat = self.transformer(feats.mean(dim=1))  # 时间维度平均
        return self.action_head(feat)
```

**改进方案C（最先进，适合超长序列）**：Mamba（状态空间模型）：
- 论文：Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2023)
- 特点：Transformer 的建模能力 + RNN 的推理效率
- 适合：如果 wukong_ai 要用 30 秒以上的历史帧（450+ 帧），Mamba 比 Transformer 高效得多

---

## 三、综合建议：wukong_ai 应该怎么做

### 3.1 架构演进路线（对应自动驾驶三代）

```
现在（第一代：纯端到端 BC）
  ↓ 添加 goal 变量
  ↓ 添加时序建模（LSTM/Transformer）
  ↓ 添加 MoE（战斗专家 + 寻路专家）
未来（第三代：端到端 + 学习的中间表示）
```

### 3.2 具体实施步骤

**Week 1（P0）**：
- [ ] DAgger 数据采集（解决 compounding error）
- [ ] 修复鼠标头（SmoothL1 + 3x 权重）

**Week 2（P1）**：
- [ ] 目标条件 BC（goal embedding）
- [ ] 在 ResNet 后加 2 层 LSTM（时序建模）

**Week 3-4（P1+）**：
- [ ] 双流 MoE 架构（战斗专家 + 寻路专家）
- [ ] Gate 网络学习软切换权重

**Week 5+（P2）**：
- [ ] Transformer 替代 LSTM（如果数据量足够）
- [ ] 拓扑地图构建（自动发现游戏世界结构）
- [ ] 分层 RL（高层规划 + 底层执行）

### 3.3 为什么时序建模会大幅改善效果

当前 wukong_ai 的根本问题：**每帧独立判断，没有上下文**。

举个具体例子：
```
当前画面：一条路往前延伸
  → 模型输出：forward（正确）

3秒后，画面：同样的路往前延伸，但左边出现岔路口
  → 模型输出：forward（还是 forward，因为它不记得3秒前没看到岔路口）
  → 正确输出：应该准备左转
```

加了时序建模后：
```
LSTM 隐藏状态记住了：「3秒前没岔路，现在出现了，目标在左边」
  → 模型输出：slow down + prepare left turn（正确）
```

**语音LLM也是同样的问题**：理解一句话需要整句话的上下文，不能只看当前音节。

---

## 四、参考资料

### 论文

| 论文 | 年份 | 关键思想 | 相关性 |
|------|------|---------|--------|
| PilotNet (NVIDIA) | 2016 | 端到端自动驾驶 | ⭐⭐ 反面教材 |
| Wayve | 2023 | 端到端 + 中间表示 | ⭐⭐⭐⭐ 正面参考 |
| Back to Basics: Revisiting ASR | 2026 | 语音识别鲁棒性 | ⭐⭐⭐ 数据采集参考 |
| Mamba | 2023 | 状态空间模型 | ⭐⭐⭐⭐ 时序建模 |
| Qwen2-Audio | 2024 | 语音大语言模型 | ⭐⭐⭐⭐ 架构参考 |

### 代码仓库

| 仓库 | Stars | 内容 |
|------|-------|------|
| QwenLM/Qwen2-Audio | 2067 | 阿里语音LLM（代表性工作） |
| erdos-project/pylot | 534 | 模块化自动驾驶平台 |
| ikostrikov/pytorch-a2c-ppo-acktr-gail | 3901 | RL 全家桶 |

### 李沐资源

- B站：跟李沐学AI（搜索「李沐 语音LLM」）
- 论文笔记：github.com/mli/paper-reading（33293 stars）
- 《动手学深度学习》：d2l.ai（中英文版）

---

*研究完成时间：2026-05-19*
*下一份研究：Diffusion Policy + 世界模型（如果 MoE+时序建模效果不够好）*
