# Wukong AI - DQN for Black Myth: Wukong

Train AI to play Black Myth: Wukong using Deep Q-Network (DQN) with Experience Replay and Target Network.

## Overview

This project uses Reinforcement Learning to train an AI agent to play Black Myth: Wukong. The agent learns combat skills through real-time interaction with the game, using reward feedback from blood bar detection.

Unlike Supervised Learning, Reinforcement Learning can update its network by itself using reward feedback, meaning we no longer need to collect our own datasets. All training data comes from the real-time interaction between the DQN network and the game.

## Project Structure

```
wukong_ai/
├── DQN_wukong_training_gpu.py       # DQN training script (TensorFlow GPU)
├── DQN_wukong_training_gpu_v2.py    # DQN training v2
├── DQN_wukong_testing_gpu.py        # DQN testing/inference script
├── DQN_tensorflow_gpu.py            # DQN network (TensorFlow)
├── DQN_pytorch.py                   # DQN network (PyTorch, incomplete)
├── PPO_pytorch.py                   # PPO algorithm (PyTorch)
├── PPO_wukong_training.py           # PPO training for Wukong
├── grabscreen.py                    # Screen capture utility
├── directkeys.py                    # Keyboard input simulation
├── getkeys.py                       # Key recording utility
├── find_blood_location.py           # Blood bar detection
├── mask_utils.py                    # Mask utilities
├── data_video_utils.py              # Video data utilities
├── demo_deque.py                    # Demo for deque usage
├── restart.py                       # Game restart utility
├── utils_test.py                    # Utility tests
├── utils/
│   ├── utils_main.py                # Main utility functions
│   ├── win32_input.py               # Win32 input handling
│   ├── wukong_win_func.py           # Wukong window functions
│   ├── wukong_yolo_pose.py          # YOLO pose detection for Wukong
│   ├── t1.py                        # Test script
│   ├── t2_yolov8.py                 # YOLOv8 test
│   ├── t3_yolov8.py                 # YOLOv8 test v3
│   └── t_demo.py                    # Demo test
└── requirement.txt                  # Python dependencies
```

## Algorithm

- **DQN** with Experience Replay and Target Network
- **State**: Game screenshots (grayscale, resized)
- **Action Space**: 6 discrete actions (attack, dodge, etc.)
- **Reward**: Based on pixel-detected blood bars (self vs boss)

## Requirements

```
tensorflow-gpu
opencv-python
numpy
pillow
pywin32
```

## Usage

1. Launch Black Myth: Wukong
2. Run the training script: `python DQN_wukong_training_gpu.py`
3. The AI will start learning through game interaction

## History

This project was originally developed for Sekiro: Shadows Die Twice and has been adapted for Black Myth: Wukong. The original Sekiro version is available at [sekiro_tensorflow](https://github.com/analoganddigital/sekiro_tensorflow).

## License

Reference: https://github.com/Sentdex/pygta5/blob/master/LICENSE
