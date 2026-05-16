"""
wukong_env.py - 黑神话：悟空 Gym风格环境
将截图、血量检测、动作执行封装为标准RL环境接口
"""
import time
import numpy as np
import cv2
from collections import deque

from config import (
    FRAME_STACK, FRAME_WIDTH, FRAME_HEIGHT, NUM_ACTIONS,
    DETECT, ACTION_SPACE
)
from env.screen_capture import ScreenCapture
from env.blood_detector import BloodDetector
from env.action_executor import ActionExecutor


class WukongEnv:
    """
    黑神话：悟空 RL环境
    
    状态空间: (FRAME_STACK, 3, FRAME_HEIGHT, FRAME_WIDTH) 归一化图像帧堆叠
              + 血量信息（拼接在MLP输入中）
    动作空间: Discrete(NUM_ACTIONS)
    奖励:     连续值，基于血量变化 + 击杀/死亡
    """
    
    def __init__(
        self,
        capture_backend=None,
        frame_stack=None,
        frame_size=None,
        reward_scale=1.0,
    ):
        self.capture = ScreenCapture(backend=capture_backend)
        self.detector = BloodDetector()
        self.executor = ActionExecutor()
        
        self.frame_stack = frame_stack or FRAME_STACK
        self.frame_w = frame_size[0] if frame_size else FRAME_WIDTH
        self.frame_h = frame_size[1] if frame_size else FRAME_HEIGHT
        
        self.reward_scale = reward_scale
        
        # 帧堆叠缓冲区
        self._frames = deque(maxlen=self.frame_stack)
        
        # 环境状态
        self._done = False
        self._step_count = 0
        self._episode_reward = 0.0
        self._prev_vitals = None
        
        # 环境元数据
        self.metadata = {
            "render.modes": ["human"],
            "video.frames_per_second": 60,
        }
    
    @property
    def observation_space_shape(self):
        """状态空间形状"""
        return (self.frame_stack * 3, self.frame_h, self.frame_w)
    
    @property
    def action_space_n(self):
        """动作空间大小"""
        return NUM_ACTIONS
    
    def reset(self):
        """
        重置环境：检测游戏状态，如果已死亡则重新开始
        
        Returns:
            observation: 初始观测
            info: dict
        """
        self._done = False
        self._step_count = 0
        self._episode_reward = 0.0
        self._frames.clear()
        self.detector.reset()
        
        # 检查是否需要重新开始游戏
        frame = self.capture.grab()
        vitals = self.detector.get_all_vitals(frame)
        
        if self.detector.is_player_dead(frame):
            print("[WukongEnv] 检测到死亡，正在重新开始...")
            self._restart_game()
            time.sleep(DETECT["restart_wait"])
            frame = self.capture.grab()
            vitals = self.detector.get_all_vitals(frame)
        
        self._prev_vitals = vitals
        
        # 填充帧堆叠
        processed = self._preprocess_frame(frame)
        for _ in range(self.frame_stack):
            self._frames.append(processed)
        
        obs = self._get_observation()
        info = self._get_info(vitals)
        
        return obs, info
    
    def step(self, action):
        """
        执行一步动作
        
        Args:
            action: int，动作ID
        
        Returns:
            observation: 新观测
            reward: float，奖励值
            terminated: bool，是否结束（死亡/击杀）
            truncated: bool，是否截断（超时）
            info: dict，附加信息
        """
        if self._done:
            print("[WukongEnv] 环境已结束，请调用 reset()")
            return self._get_observation(), 0.0, True, False, {}
        
        # 执行动作
        action_name = self.executor.execute(action)
        
        # 等待一帧
        time.sleep(0.02)  # ~50fps
        
        # 获取新帧
        frame = self.capture.grab()
        vitals = self.detector.get_all_vitals(frame)
        
        # 计算奖励
        reward = self._compute_reward(vitals, action_name)
        
        # 检查终止条件
        terminated = False
        truncated = False
        
        if self.detector.is_player_dead(frame):
            terminated = True
            reward += -5.0 * self.reward_scale  # 死亡惩罚
            print(f"[WukongEnv] 死亡！步数={self._step_count}, 累计奖励={self._episode_reward:.2f}")
        
        if self.detector.is_boss_dead(frame):
            terminated = True
            reward += 20.0 * self.reward_scale  # 击杀奖励
            print(f"[WukongEnv] 击杀Boss！步数={self._step_count}, 累计奖励={self._episode_reward:.2f}")
        
        # 更新帧堆叠
        processed = self._preprocess_frame(frame)
        self._frames.append(processed)
        
        # 更新状态
        self._step_count += 1
        self._episode_reward += reward
        self._prev_vitals = vitals
        self._done = terminated or truncated
        
        obs = self._get_observation()
        info = self._get_info(vitals)
        info["action_name"] = action_name
        info["reward_detail"] = reward
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, vitals, action_name):
        """
        计算奖励 - 连续值设计
        
        核心原则：
        1. 伤害Boss → 正奖励（按比例）
        2. 被Boss打 → 负奖励（按比例）
        3. 攻击消耗气力 → 微小负奖励（避免无意义攻击）
        4. 闪避成功 → 小正奖励（减少伤害时）
        """
        reward = 0.0
        
        # Boss血量变化：给Boss造成伤害=正奖励
        if vitals["boss_hp_delta"] < -0.001:
            reward += abs(vitals["boss_hp_delta"]) * 50.0 * self.reward_scale
        
        # 玩家血量变化：受伤=负奖励
        if vitals["player_hp_delta"] < -0.001:
            reward += vitals["player_hp_delta"] * 30.0 * self.reward_scale
        
        # 气力消耗：微小惩罚避免无意义操作
        if vitals["player_stamina"] < 0.2 and action_name in ("attack", "dodge", "heavy_attack"):
            reward -= 0.1 * self.reward_scale
        
        # 每步存活微小奖励
        reward += 0.01 * self.reward_scale
        
        return reward
    
    def _preprocess_frame(self, frame):
        """
        预处理帧：缩放 + 归一化
        
        Args:
            frame: BGR np.ndarray (H, W, 3)
        
        Returns:
            np.ndarray: (3, H, W) 归一化到 [0, 1]
        """
        # 缩放
        resized = cv2.resize(frame, (self.frame_w, self.frame_h))
        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # 归一化
        normalized = rgb.astype(np.float32) / 255.0
        # (H, W, 3) -> (3, H, W)
        return normalized.transpose(2, 0, 1)
    
    def _get_observation(self):
        """
        获取当前观测：帧堆叠
        
        Returns:
            np.ndarray: (FRAME_STACK * 3, H, W)
        """
        return np.concatenate(list(self._frames), axis=0)
    
    def _get_info(self, vitals):
        """获取附加信息"""
        return {
            "step": self._step_count,
            "episode_reward": self._episode_reward,
            "player_hp": vitals["player_hp"],
            "boss_hp": vitals["boss_hp"],
            "player_stamina": vitals["player_stamina"],
        }
    
    def _restart_game(self):
        """重新开始游戏（死亡后）"""
        time.sleep(DETECT["death_wait"])
        # 按确认键重新开始
        for key in DETECT["restart_keys"]:
            self.executor.execute_raw([key], duration_ms=200)
            time.sleep(0.5)
    
    def get_visual_obs(self):
        """
        获取带血量信息的完整观测
        用于模型推理时，图像帧 + 血量数值
        
        Returns:
            dict: {
                "frames": np.ndarray (FRAME_STACK*3, H, W),
                "blood_info": np.ndarray [player_hp, boss_hp, stamina],
            }
        """
        obs = self._get_observation()
        if self._prev_vitals:
            blood = np.array([
                self._prev_vitals["player_hp"],
                self._prev_vitals["boss_hp"],
                self._prev_vitals["player_stamina"],
            ], dtype=np.float32)
        else:
            blood = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        return {"frames": obs, "blood_info": blood}
    
    def close(self):
        """关闭环境，释放资源"""
        self.capture.release()
        print("[WukongEnv] 环境已关闭")
