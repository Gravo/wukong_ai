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
## 2026-05-20
- 完成了什么：
- 遇到了什么问题：
- 明天计划：
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

**最后更新**: 2026-05-19 23:45  
**下次更新**: 完成任務1（Goal-Conditioned BC）后
