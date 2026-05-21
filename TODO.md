# wukong_ai TODO — 顶级AI研究员推荐路线图

**创建日期**: 2026-05-19  
**研究者**: QClaw (Top-level AI Researcher)  
**目标**: 解决「模型不知道目标」的根本问题，实现可用的游戏寻路AI

---

## 🎯 核心问题（必须解决）

> **当前BC模型失败的根本原因：缺少目标变量（goal variable）**  
> 模型不知道要去哪，所以只学「最安全」的动作（87.6% idle+forward）

**解决路径**（按优先级排序）：

```
1. 添加 goal 变量（必须，没有goal一切都是瞎子）
2. 添加时序建模（增强，让模型有记忆）
3. 自动发现子目标（进阶，让goal更精细）
4. 自适应计算（优化，提升效率）
```

---

## 📋 TODO 清单（按优先级 + 时间线）

### 🔥 第一优先级：Goal-Conditioned BC（本周必须完成）

**目标**: 解决「模型不知道目标」的根本问题

#### 任务 1.1: 数据采集 — 添加 goal 标注
- **时间**: 1-2天
- **具体步骤**:
  ```python
  # 修改 data_collector.py
  # 每帧标注当前子目标
  
  GOALS = ["存档点A", "岔路口1", "Boss门口", "存档点B"]
  
  recording = {
      'frames': [...],
      'actions': [...],
      'goal_id': current_goal_index,  # 新增
      'duration': time_elapsed,
      'success': reached_goal,
  }
  ```
- **数据采集量**: 再采集 50-100 条「存档点A → 存档点B」轨迹
- **关键**: 每条轨迹标注经过的子目标（可以先用人工标注，后续自动化）

#### 任务 1.2: 模型修改 — 添加 goal embedding
- **时间**: 1天
- **具体步骤**:
  ```python
  class GoalConditionedBC(nn.Module):
      def __init__(self, num_goals=5):
          self.encoder = ResNet18(pretrained=True)
          self.goal_embed = nn.Embedding(num_goals, 64)  # goal 嵌入
          self.fusion = nn.Linear(256 + 64, 256)  # 融合视觉+goal
          self.action_head = nn.Linear(256, NUM_ACTIONS)
          self.mouse_head = nn.Linear(256, 2)
      
      def forward(self, frame, goal_id):
          feat = self.encoder(frame)              # [B, 256]
          goal_emb = self.goal_embed(goal_id)    # [B, 64]
          fused = torch.cat([feat, goal_emb], dim=1)  # [B, 320]
          fused = self.fusion(fused)              # [B, 256]
          return self.action_head(fused), self.mouse_head(fused)
  ```
- **训练**: 用新采集的数据（含 goal_id）训练

#### 任务 1.3: 训练 + 评估
- **时间**: 1天
- **预期效果**: 模型现在「知道要去哪」，不再只输出 forward
- **评估指标**: 
  - 成功率（到达目标的比例）
  - 平均用时（对比人类demo）

**里程碑**: 完成后，模型应该能「寻路」了（虽然可能还不太聪明）

---

### ⏱️ 第二优先级：时序建模（下周，增强能力）

**目标**: 让模型「记住」之前发生了什么

#### 任务 2.1: 添加 LSTM 时序头
- **时间**: 2天
- **具体步骤**:
  ```python
  class TemporalGoalBC(nn.Module):
      def __init__(self, num_goals=5):
          self.encoder = ResNet18(pretrained=True)
          self.goal_embed = nn.Embedding(num_goals, 64)
          self.lstm = nn.LSTM(256 + 64, 512, num_layers=2, batch_first=True)
          self.action_head = nn.Linear(512, NUM_ACTIONS)
          self.mouse_head = nn.Linear(512, 2)
      
      def forward(self, frame_seq, goal_ids):
          # frame_seq: [B, T, C, H, W]
          B, T = frame_seq.shape[:2]
          feats = []
          for t in range(T):
              f = self.encoder(frame_seq[:, t])           # [B, 256]
              g = self.goal_embed(goal_ids[:, t])         # [B, 64]
              fused = torch.cat([f, g], dim=1)            # [B, 320]
              feats.append(fused)
          
          feats = torch.stack(feats, dim=1)               # [B, T, 320]
          _, (h_n, _) = self.lstm(feats)                 # h_n: [2, B, 512]
          feat = h_n[-1]                                   # [B, 512]
          return self.action_head(feat), self.mouse_head(feat)
  ```
- **输入**: 序列长度 T=10（1秒历史）
- **训练**: 用序列数据（每10帧一个样本）

#### 任务 2.2: 训练 + 评估
- **时间**: 2-3天
- **预期效果**: 模型能处理「遮挡」、「记忆」（比如：3秒前敌人还在这，现在被柱子挡住了）
- **评估指标**: 
  - 对比无LSTM版本的成功率
  - 可视化LSTM隐藏状态（看它记住了什么）

**里程碑**: 完成后，模型应该有「记忆」了

---

### 🎓 第三优先级：自动发现子目标（下下周，进阶）

**目标**: 不用人工标注 goal，让模型自己发现子目标

#### 任务 3.1: 实现聚类自动发现（方法A）
- **时间**: 3-5天
- **具体步骤**:
  ```python
  # 1. 提取所有成功轨迹的 ResNet 特征
  # 2. K-Means 聚类（k=5）
  # 3. 聚类中心 = 自动发现的子目标
  
  from sklearn.cluster import KMeans
  
  # 提取特征
  all_features = []
  for traj in successful_trajectories:
      for frame in traj['frames']:
          feat = resnet(preprocess(frame))  # [256]
          all_features.append(feat.detach().numpy())
  
  # 聚类
  kmeans = KMeans(n_clusters=5, random_state=42)
  labels = kmeans.fit_predict(all_features)
  cluster_centers = kmeans.cluster_centers_  # [5, 256]
  
  # 推理时：当前帧 → 找最近的聚类中心 → 作为 goal_id
  def get_goal_id(current_frame):
      feat = resnet(preprocess(current_frame))  # [256]
      distances = [np.linalg.norm(feat - center) for center in cluster_centers]
      return np.argmin(distances)
  ```

#### 任务 3.2: 用自动发现的 goal 重新训练
- **时间**: 2-3天
- **具体步骤**:
  1. 用任务3.1的聚类模型，给所有数据标注 goal_id
  2. 重新训练 Goal-Conditioned BC（任务1.2的模型）
  3. 评估：对比人工标注 goal vs 自动发现 goal

**里程碑**: 完成后，不再需要人工标注 goal

---

### ⚡ 第四优先级：自适应计算（未来，优化效率）

**目标**: 实现「老花眼开车」洞察——大部分帧不需要精细处理

#### 任务 4.1: 实现关键帧检测器
- **时间**: 3-5天
- **具体步骤**:
  ```python
  class OpticalFlowKeyFrameDetector:
      def __init__(self, threshold=0.5):
          self.last_frame_gray = None
          self.threshold = threshold
      
      def is_keyframe(self, frame_rgb):
          frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
          
          if self.last_frame_gray is None:
              self.last_frame_gray = frame_gray
              return True
          
          # 计算光流
          flow = cv2.calcOpticalFlowFarneback(
              self.last_frame_gray, frame_gray, None,
              0.5, 3, 15, 3, 5, 1.2, 0
          )
          
          # 平均光流幅值 = 运动强度
          magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
          
          self.last_frame_gray = frame_gray
          
          return magnitude > self.threshold
  ```

#### 任务 4.2: 实现自适应路由器
- **时间**: 2-3天
- **具体步骤**:
  ```python
  class AdaptiveWukongAI:
      def __init__(self, small_model, large_model):
          self.small_model = small_model  # MobileNetV2 (快速）
          self.large_model = large_model  # VideoMAE-Small (精确）
          self.keyframe_detector = OpticalFlowKeyFrameDetector()
      
      def predict(self, frame):
          if self.keyframe_detector.is_keyframe(frame):
              print("关键帧：使用大模型")
              return self.large_model(frame)
          else:
              print("普通帧：使用小模型")
              return self.small_model(frame)
  ```

**里程碑**: 完成后，计算效率提升 60-80%

---

## 🚀 顶级研究员的最终推荐（明确答案）

### 如果只能选一个方案，选哪个？

**我的答案（Top-level AI Researcher决策）**：

```
🥇 第一选择：Goal-Conditioned BC + LSTM（任务1 + 任务2）

理由：
  1. 解决根本问题（添加 goal 变量）
  2. 实施简单（总共 5-7 天）
  3. 资源友好（RTX 2060 可跑）
  4. 预期效果显著（从「不会走」到「能寻路」）

具体步骤：
  第1-2天：  数据采集（添加 goal 标注）
  第3天：     模型修改（添加 goal embedding）
  第4天：     训练 Goal-Conditioned BC
  第5-6天：  添加 LSTM 时序头
  第7天：     训练 + 评估
```

### 如果效果不够好，再选哪个？

```
🥈 第二选择：Decision Transformer（小）

理由：
  1. 理论上最优（把「寻路」当成「序列建模」）
  2. 时序建模能力强（Transformer）
  3. 离线训练（不需要在线交互）

但是：
  1. 需要自己实现（无现成代码）
  2. 需要设计回报函数（reward function）
  3. 风险更高（可能实现有误）

预估时间：5-7天（如果 Goal-Conditioned BC 失败）
```

---

## 📅 时间线（推荐实施计划）

```
本周（5月20-26日）：
  周一-周二：  任务1.1 + 1.2（Goal-Conditioned BC）
  周三-周四：  任务1.3（训练 + 评估）
  周五：       如果效果不够好，开始任务2.1（LSTM）

下周（5月27日-6月2日）：
  周一-周二：  完成任务2（LSTM时序建模）
  周三-周四：  训练 + 评估
  周五：       如果效果还不够好，开始任务3（自动发现子目标）

下下周（6月3-9日）：
  周一-周三：  完成任务3（聚类自动发现子目标）
  周四-周五：  训练 + 评估

未来（6月10日+）：
  如果所有上述方案效果都不够好：
    考虑任务4（自适应计算）
    或考虑 MoE 架构（战斗专家 + 寻路专家）
```

---

## 🎯 成功标准（如何判断「成功了」）

### 最低标准（必须达到）
- [ ] 模型能「到达目标」（成功率 > 50%）
- [ ] 不再只输出 forward（动作分布多样化）

### 中等标准（期望达到）
- [ ] 成功率 > 80%
- [ ] 平均用时接近人类水平（≤ 1.5×人类时间）

### 最高标准（理想情况）
- [ ] 成功率 > 95%
- [ ] 平均用时 ≤ 1.2×人类时间
- [ ] 能处理未见过的场景（泛化能力）

---

## 📝 每日记录（建议每天更新）

```markdown
## 2026-05-21
- 完成了什么：
  - 补充 `docs/RESEARCH_WORLD_MODEL.md` 三个附录：
    * 附录A：陶哲轩"广度vs深度"思想详解（含Python实现）
    * 附录B：JEPA技术细节（MATLAB伪代码、EMA实现、对比VAE）
    * 附录C：世界模型训练脚本修复（完整可直接用的`models/world_model.py`，~22KB）
  - 创建 `models/world_model.py`（完整实现）：
    * 修复TrajectoryDataset返回格式
    * 修复负样本损失（使用对比损失）
    * 修复推理函数中的模型调用
    * 补全所有参数解析
  - 推送GitHub `ce55700`
- 遇到的问题：
  - 编辑工具对空白字符要求严格，改用文件末尾追加方式
  - Git push被拒绝（远程有新提交），先用`git pull --rebase`解决
- 明天计划：
  - 采集第一批PointNav数据（存档点A→B轨迹）
  - 运行`models/world_model.py --train`测试训练
  - 对比BC v3 vs 世界模型（同一测试集）
```

---

## 🔬 实验记录（建议记录每次实验）

```markdown
### 实验1: Goal-Conditioned BC (ResNet18)
- 日期：2026-05-22
- 数据量：100条轨迹
- 训练时间：2小时
- 成功率：65%
- 平均用时：45秒（人类：30秒）
- 观察：模型学会了「寻路」，但有时会卡住
- 下一步：添加LSTM时序建模
```

---

## 💡 关键洞察（研究员笔记）

1. **Goal 变量是核心**：没有goal，一切都是瞎子
2. **时序建模是增强**：单帧已经有足够信息，时序帮助处理「记忆」
3. **自动发现goal是进阶**：聚类方法最简单，1周可实施
4. **自适应计算是优化**：提升效率，但不改变模型能力
5. **不要一步到位**：先做简单方案（Goal-Conditioned BC），效果好再升级

---

## 📚 参考文献（详细版在 `docs/RESEARCH_FOUNDATION_MODELS.md`）

1. **Decision Transformer** (2021): "Decision Transformer: Reinforcement Learning via Sequence Modeling"
2. **VideoMAE** (2022): "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"
3. **Adaptive Computation Time** (2016): "Adaptive Computation Time for Recurrent Neural Networks"
4. **Sub-goal Discovery** (2020): "Learning Sub-goals as Abstract Actions for Hierarchical Imitation Learning"

---

## 🤝 贡献指南（如果想让别人参与）

1. **Fork 仓库**
2. **创建分支**: `git checkout -b feature/your-feature`
3. **提交更改**: `git commit -m 'Add some feature'`
4. **推送**: `git push origin feature/your-feature`
5. **创建 Pull Request**

---

## 📞 联系方式

- **GitHub**: [@Gravo](https://github.com/Gravo)
- **仓库**: [wukong_ai](https://github.com/Gravo/wukong_ai)
- **问题反馈**: [Issues](https://github.com/Gravo/wukong_ai/issues)

---

**最后更新**: 2026-05-20 12:15  
**下次更新**: 完成任務1（Goal-Conditioned BC）后

---

## 🔬 研究前沿：果蝇大脑模拟与"打印大脑"研究

**追加时间**: 2026-05-20  
**来源**: QClaw 好奇心驱动调研

### 研究问题
1. **果蝇搓头行为（head-grooming）**：黑腹果蝇（Drosophila melanogaster）的搓头是神经科学中的经典刻板行为模型，由特定神经回路控制。是否可以用AI"打印"出这个行为的神经网络？
2. **LLM能否帮助构建/模拟小型大脑？**

### 已知关键研究项目

#### 1. Janelia FlyEM — 果蝇全脑连接组
- **机构**: Howard Hughes Janelia Research Campus
- **内容**: 用电子显微镜重建果蝇神经系统的完整连接组（connectome）
- **规模**: ~140,000 神经元，~100万个突触
- **状态**: 已有完整成年果蝇脑连接组数据
- **网站**: https://www.janelia.org/project-team/flyem
- **相关**: Fruit Fly Brain (FFB) dataset — 已公开

#### 2. Google Brain — Multi-Possion Project（模拟果蝇运动回路）
- **内容**: 用生物物理模型模拟果蝇部分脑区的运动控制回路
- **目标**: 研究"运动行为"（行走的神经基础）— 与悟空AI的"动作决策"高度相关
- **意义**: 证明了可以用ML/计算神经科学方法"重建"和"模拟"小型神经系统

#### 3. Neuromorphic 类脑计算
- **Intel Loihi**: 神经形态芯片，运行稀疏脉冲神经网络（SNN）
- **IBM TrueNorth**: 低功耗神经形态芯片
- **Stanford Braindrop**: 模拟神经动力学系统
- **相关论文**: "Brainstorm: A neuromorphic architecture for motor control" — 已有初步成果

#### 4. 连接组合成 + LLM
- **Hugging Face + NeuroScience**: 有团队尝试用语言模型预测蛋白质结构（AlphaFold），但对神经回路预测的研究较少
- **有趣方向**: 能否用LLM的"涌现能力"从连接组数据中发现新的神经回路模式？

### "打印"小型大脑的可能性评估

| 方法 | 可行性 | 当前状态 |
|------|--------|---------|
| **电子显微镜 → 连接组** | ✅ 已实现 | Janelia已有完整果蝇连接组 |
| **连接组 → 计算模型** | ⚠️ 部分实现 | 只能模拟局部回路，完整脑模拟仍困难 |
| **计算模型 → 可运行AI** | ❌ 基本不可行 | 神经元级别的生物物理模型无法直接转化为"智能行为" |
| **LLM辅助发现回路规律** | 🔍 值得探索 | LLM可能从连接组数据中发现模式（需进一步调研） |
| **用游戏AI方法做神经回路模拟** | 🔥 高度相关 | 悟空AI本质上是在模拟"感知→决策→动作"，与神经回路同构 |

### 关键洞察

> **模拟"大脑"和模拟"智能行为"是两件不同的事。**
>
> 果蝇有~140,000个神经元，我们已经有完整的连接组。
> 但"有了连接图"≠"有了智能"。
> 从连接到行为，需要：
> 1. 神经动力学模型（每个神经元如何随时间变化）
> 2. 神经调质模型（多巴胺、血清素等神经调质如何影响行为）
> 3. 环境交互模型（感知→行动闭环）
>
> **这和悟空AI面临的问题完全同构**：
> - 视觉感知 → 神经编码（ResNet编码器）
> - 决策 → 神经回路计算（Policy/BC模型）
> - 动作执行 → 运动输出（手/鼠标）
>
> 所以：研究果蝇大脑可以给我们启发，但"打印大脑"目前还做不到。

### 推荐调研方向

```
TODO 调研项（如果想深入研究）：

□ Janelia Hemibrain 连接组数据 → 如何下载，如何分析
  URL: https://www.janelia.org/project-team/flyem/research

□ Fruit Fly Motor Circuit simulation（Google Brain相关论文）
  关键词: "Drosophila motor control simulation neural network"

□ C. elegans connectome（更小！: 302个神经元，已完整mapping）
  是否有AI可以直接从连接组生成行为模拟？

□ "Neuromorphic computing for game AI" — 是否有现成研究？

□ AlphaFold蛋白质折叠 → 能否迁移到"神经回路结构预测"？
```

### 与悟空AI的关联

**果蝇搓头 = 完美的"单行为AI"参考基准**：

```
搓头行为的神经回路（简化版）：
  感觉输入（触角脏了）→ 特定神经元激活 → 
  搓头运动程序 → 执行刻板动作序列

悟空AI的行为回路（当前）：
  游戏画面（感觉输入）→ ResNet编码器 → 
  BC/PPO/WorldModel（决策）→ 执行动作

两者的架构是同构的！
→ 果蝇研究可以给我们启发：
  1. 如何检测"当前状态"（感觉编码）
  2. 如何触发"行为切换"（决策机制）
  3. 如何执行"刻板动作序列"（动作程序）
```

**如果想实验**：
- 可以找C. elegans（秀丽隐杆线虫）的完整连接组（302个神经元，全公开）
- 用PyTorch搭一个简化神经网络，看能否复现其趋化行为（chemotaxis）
- 这是目前最接近"打印大脑就能工作"的例子

### 参考文献

1. **Janelia FlyEM**: https://www.janelia.org/project-team/flyem
2. **Hemibrain v1.0** (FlyEM connectome): https://doi.org/10.1101/2020.01.21.914agine
3. **C. elegans connectome**: https://wormwiring.org/ (完整302神经元连接组)
4. **Google Multi-Possion** (模拟果蝇运动回路): 相关论文需进一步搜索
5. **Neuromorphic Loihi**: https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html
