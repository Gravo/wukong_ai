# 🐒 wukong_ai

Black Myth: Wukong (黑神话：悟空) combat AI powered by Reinforcement Learning.

Currently training against **虎先锋 (Tiger Vanguard)**.

## Architecture

```
wukong_ai/
├── config.py                    # 集中配置（超参数、窗口坐标、动作空间）
├── requirements.txt
├── env/
│   ├── wukong_env.py           # Gym风格RL环境
│   ├── screen_capture.py       # 高速截图 (dxcam/mss/win32)
│   ├── blood_detector.py       # HSV色域血量检测
│   └── action_executor.py      # pydirectinput动作执行
├── models/
│   ├── resnet_encoder.py       # ResNet18视觉编码器
│   └── ppo_agent.py            # PPO Actor-Critic
├── training/
│   ├── train_combat.py         # 战斗训练主脚本
│   └── data_collector.py       # 人类Demo数据采集
├── pathfinding/
│   └── behavior_clone.py       # 行为克隆（寻路阶段）
├── utils_new/
│   ├── replay_buffer.py        # Rollout缓冲 + 优先经验回放
│   └── logger.py               # TensorBoard + 文件日志
└── old/                        # 旧版代码归档（DQN/PPO buggy versions）
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Calibrate blood detection

Before training, you need to calibrate the blood bar positions and HSV color ranges for your screen resolution. Edit `config.py`:

- `GAME_REGION`: your game window resolution
- `BLOOD_REGION`: blood bar pixel coordinates
- `HSV_RANGES`: blood bar color ranges

### 3. Collect demo data (for pathfinding)

Launch the game and play through the path to Tiger Vanguard while recording:

```bash
python -m training.data_collector --mode pathfinding --episodes 5
```

### 4. Train behavior cloning (pathfinding)

```bash
python -m pathfinding.behavior_clone --data-dir pathfinding_data
```

### 5. Train PPO (combat)

```bash
python -m training.train_combat train
```

### 6. Evaluate

```bash
python -m training.train_combat eval --model checkpoints/best_model.pt
```

## Technical Highlights

| Feature | Old (DQN) | New (PPO) |
|---------|-----------|-----------|
| Algorithm | DQN (TF1, buggy) | PPO (PyTorch, correct) |
| State | Single grayscale frame | 4-frame stack + blood values |
| Encoder | 2-conv CNN | ResNet18 (ImageNet pretrained) |
| Blood detection | Grayscale pixel counting | HSV color space segmentation |
| Screen capture | win32gui (~30fps) | dxcam (~120fps) |
| Action input | SendInput | pydirectinput (DirectX compatible) |
| Reward | Hardcoded 6-level discrete | Continuous + normalized |
| Pathfinding | Key replay (open-loop) | Behavior cloning + RL (closed-loop) |
| Replay buffer | 2000 (too small) | 100,000 with prioritized sampling |

## Action Space

| ID | Action | Keys |
|----|--------|------|
| 0 | Idle | - |
| 1 | Attack | LMB |
| 2 | Heavy Attack | LMB x4 |
| 3 | Dodge | Space |
| 4 | Move Forward | W |
| 5 | Move Right | D |
| 6 | Move Left | A |
| 7 | Dodge + Attack | Space + LMB |
| 8 | Lock On | V |
| 9 | Heal (Gourd) | R |

## License

MIT (original project by analogandigital, rewritten by Gravo)
