"""
action_executor.py - 游戏动作执行模块
使用 pydirectinput 替代旧版 SendInput，解决DirectX游戏输入问题
"""
import time
import numpy as np
from config import ACTION_SPACE, KEY_MAP, NUM_ACTIONS

class ActionExecutor:
    """游戏动作执行器"""
    
    def __init__(self):
        import pydirectinput
        self._pydi = pydirectinput
        self._last_action_time = 0
        self._current_action = 0  # 当前执行的动作ID
        self._action_cooldown = 0.05  # 动作间最小间隔（秒）
    
    def execute(self, action_id):
        """
        执行一个动作
        
        Args:
            action_id: int，动作编号（0 ~ NUM_ACTIONS-1）
        
        Returns:
            str: 执行的动作名称
        """
        if not 0 <= action_id < NUM_ACTIONS:
            print(f"[ActionExecutor] 无效动作ID: {action_id}")
            return "invalid"
        
        # 冷却检查
        now = time.time()
        if now - self._last_action_time < self._action_cooldown:
            time.sleep(self._action_cooldown - (now - self._last_action_time))
        
        action_name, keys, durations = ACTION_SPACE[action_id]
        self._current_action = action_id
        
        if not keys:
            # idle动作，不做任何事
            self._last_action_time = time.time()
            return action_name
        
        # 执行按键序列
        if isinstance(durations, list):
            # 组合动作：按顺序按下多个键，每个键有不同的持续时间
            for key, dur in zip(keys, durations):
                self._press_key(key, dur)
        else:
            # 单键或同时按下的键
            if durations > 0:
                for key in keys:
                    self._press_key(key, durations)
            else:
                # duration=0，瞬间按下释放
                for key in keys:
                    self._tap_key(key)
        
        self._last_action_time = time.time()
        return action_name
    
    def _press_key(self, key, duration_ms):
        """
        按下并保持一个键
        
        Args:
            key: str，按键名称（如 "w", "space", "j"）
            duration_ms: int/float，按下持续时间（毫秒）
        """
        # J键映射为鼠标左键点击（黑神话悟空用鼠标攻击）
        if key == "j":
            self._pydi.mouseDown(button='left')
            time.sleep(duration_ms / 1000.0)
            self._pydi.mouseUp(button='left')
        elif key == "k":
            self._pydi.mouseDown(button='right')
            time.sleep(duration_ms / 1000.0)
            self._pydi.mouseUp(button='right')
        else:
            self._pydi.keyDown(key)
            time.sleep(duration_ms / 1000.0)
            self._pydi.keyUp(key)
    
    def _tap_key(self, key):
        """快速点击一个键"""
        if key == "j":
            self._pydi.click(button='left')
        elif key == "k":
            self._pydi.click(button='right')
        else:
            self._pydi.press(key)
    
    def execute_raw(self, keys, duration_ms=100):
        """
        直接执行自定义按键序列（用于特殊操作，如重新开始）
        
        Args:
            keys: list[str]，按键列表
            duration_ms: int，每个键的持续时间
        """
        for key in keys:
            self._press_key(key, duration_ms)
            time.sleep(0.05)  # 按键间隔
    
    def get_action_name(self, action_id):
        """获取动作名称"""
        if 0 <= action_id < NUM_ACTIONS:
            return ACTION_SPACE[action_id][0]
        return "invalid"
    
    def get_action_mask(self, player_hp=1.0, boss_hp=1.0):
        """
        根据当前状态生成动作掩码
        某些状态下某些动作没有意义，可以屏蔽以加速学习
        
        Returns:
            np.ndarray: shape=(NUM_ACTIONS,)，1.0=允许，0.0=禁止
        """
        mask = np.ones(NUM_ACTIONS, dtype=np.float32)
        
        # 血量极低时，增加回血动作的权重（但不强制，留给RL决策）
        # Boss已死时，禁止攻击动作
        if boss_hp < 0.02:
            attack_actions = [1, 2, 7]  # attack, heavy_attack, dodge_attack
            for a in attack_actions:
                mask[a] = 0.0
        
        return mask


# 全局单例
_executor_instance = None

def get_executor():
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = ActionExecutor()
    return _executor_instance
