"""
blood_detector.py - 基于HSV色域的血量检测模块
替代旧版灰度像素计数方案，更鲁棒更精确
"""
import cv2
import numpy as np
from config import BLOOD_REGION, HSV_RANGES

class BloodDetector:
    """血量检测器 - 使用HSV色域分割"""
    
    def __init__(self):
        self.regions = BLOOD_REGION
        self.hsv_ranges = HSV_RANGES
        # 缓存上一帧的血量用于计算差值
        self._prev_player_hp = 1.0
        self._prev_boss_hp = 1.0
        self._prev_stamina = 1.0
    
    def detect_hp(self, frame, region_key):
        """
        检测指定区域的血量百分比
        
        Args:
            frame: BGR格式的完整游戏画面
            region_key: "player_hp" | "boss_hp" | "player_stamina"
        
        Returns:
            float: 血量百分比 [0.0, 1.0]
        """
        if region_key not in self.regions:
            return 0.0
        
        region = self.regions[region_key]
        
        # 裁剪血条区域
        x, y, w, h = region["left"], region["top"], region["width"], region["height"]
        blood_bar = frame[y:y+h, x:x+w]
        
        if blood_bar.size == 0:
            return 0.0
        
        # 转HSV
        hsv = cv2.cvtColor(blood_bar, cv2.COLOR_BGR2HSV)
        
        # 获取HSV范围
        hsv_range = self.hsv_ranges.get(region_key)
        if hsv_range is None:
            # 没有配置HSV范围，回退到亮度检测
            return self._detect_by_brightness(blood_bar)
        
        # 创建红色掩码（红色在HSV中跨越0度，需要两段）
        lower1 = np.array(hsv_range["lower"])
        upper1 = np.array(hsv_range["upper"])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        
        if "lower2" in hsv_range:
            lower2 = np.array(hsv_range["lower2"])
            upper2 = np.array(hsv_range["upper2"])
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = mask1
        
        # 计算血量比例：有颜色的像素数 / 总像素数
        blood_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        hp_ratio = blood_pixels / total_pixels if total_pixels > 0 else 0.0
        
        return float(np.clip(hp_ratio, 0.0, 1.0))
    
    def _detect_by_brightness(self, blood_bar):
        """
        亮度回退检测法
        当HSV范围未配置时使用，基于血条区域的亮度分布
        血条有血的部分较亮，空的部分较暗
        """
        gray = cv2.cvtColor(blood_bar, cv2.COLOR_BGR2GRAY)
        # 使用Otsu自动阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright_pixels = cv2.countNonZero(binary)
        total_pixels = binary.shape[0] * binary.shape[1]
        return float(bright_pixels / total_pixels) if total_pixels > 0 else 0.0
    
    def get_all_vitals(self, frame):
        """
        一次获取所有生命体征
        
        Returns:
            dict: {
                "player_hp": float,
                "boss_hp": float,
                "player_stamina": float,
                "player_hp_delta": float,  # 相对上一帧的变化
                "boss_hp_delta": float,
            }
        """
        player_hp = self.detect_hp(frame, "player_hp")
        boss_hp = self.detect_hp(frame, "boss_hp")
        stamina = self.detect_hp(frame, "player_stamina")
        
        # 计算变化量
        player_delta = player_hp - self._prev_player_hp
        boss_delta = boss_hp - self._prev_boss_hp
        
        # 更新缓存
        self._prev_player_hp = player_hp
        self._prev_boss_hp = boss_hp
        self._prev_stamina = stamina
        
        return {
            "player_hp": player_hp,
            "boss_hp": boss_hp,
            "player_stamina": stamina,
            "player_hp_delta": player_delta,
            "boss_hp_delta": boss_delta,
        }
    
    def is_player_dead(self, frame, threshold=0.02):
        """判断玩家是否死亡"""
        hp = self.detect_hp(frame, "player_hp")
        return hp < threshold
    
    def is_boss_dead(self, frame, threshold=0.02):
        """判断Boss是否被击杀"""
        hp = self.detect_hp(frame, "boss_hp")
        return hp < threshold
    
    def calibrate(self, frame):
        """
        校准血条位置和颜色范围
        在游戏画面已知状态下（如满血、空血），自动调整HSV范围
        
        使用方式：
        1. 在满血状态下调用 calibrate(frame, "player_hp", "full")
        2. 在空血/残血状态下调用 calibrate(frame, "player_hp", "low")
        """
        print("[BloodDetector] 校准模式 - 请确保当前血条状态正确")
        for region_key in self.regions:
            hp = self.detect_hp(frame, region_key)
            print(f"  {region_key}: {hp:.2%}")
        return self.get_all_vitals(frame)
    
    def reset(self):
        """重置缓存"""
        self._prev_player_hp = 1.0
        self._prev_boss_hp = 1.0
        self._prev_stamina = 1.0


# 全局单例
_detector_instance = None

def get_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = BloodDetector()
    return _detector_instance
