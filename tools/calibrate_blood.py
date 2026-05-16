"""
calibrate_blood.py - 血量检测校准工具
交互式调整血条区域坐标和HSV颜色范围

用法:
    python calibrate_blood.py capture          # 截取游戏画面并保存
    python calibrate_blood.py visualize [img]  # 可视化当前血条区域和HSV掩码
    python calibrate_blood.py tune [img]       # 交互式调整HSV参数
    python calibrate_blood.py test [img]       # 快速测试当前配置的检测效果
    python calibrate_blood.py scan [img]       # 扫描血条区域，自动推荐HSV范围

如果未指定图片路径，默认使用最新截取的画面。
"""
import sys
import os
import json
import time
from datetime import datetime

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

# 导入项目配置
from config import BLOOD_REGION, HSV_RANGES, GAME_REGION, GAME_WINDOW_TITLE
from env.blood_detector import BloodDetector
from env.screen_capture import ScreenCapture


CAPTURE_DIR = os.path.join(PROJECT_ROOT, "calibration_captures")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def capture_game():
    """截取游戏画面"""
    ensure_dir(CAPTURE_DIR)
    print("[Capture] 正在截取游戏画面...")
    
    cap = ScreenCapture()
    frame = cap.capture()
    cap.release()
    
    if frame is None:
        print("[ERROR] 截取失败！请确认游戏窗口已打开。")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(CAPTURE_DIR, f"game_{timestamp}.png")
    cv2.imwrite(filepath, frame)
    print(f"[Capture] 已保存: {filepath}")
    print(f"  画面尺寸: {frame.shape[1]}x{frame.shape[0]}")
    return filepath


def get_latest_capture():
    """获取最新的截图"""
    if not os.path.exists(CAPTURE_DIR):
        return None
    files = sorted([f for f in os.listdir(CAPTURE_DIR) if f.endswith('.png')])
    if not files:
        return None
    return os.path.join(CAPTURE_DIR, files[-1])


def load_image(path=None):
    """加载图片"""
    if path is None:
        path = get_latest_capture()
        if path is None:
            print("[ERROR] 没有找到截图。请先运行: python calibrate_blood.py capture")
            return None
        print(f"[Load] 使用最新截图: {path}")
    
    if not os.path.exists(path):
        print(f"[ERROR] 文件不存在: {path}")
        return None
    
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] 无法读取图片: {path}")
        return None
    
    print(f"[Load] 图片尺寸: {frame.shape[1]}x{frame.shape[0]}")
    return frame


def draw_regions(frame):
    """在画面上标注血条区域"""
    annotated = frame.copy()
    colors = {
        "player_hp": (0, 255, 0),      # 绿色
        "boss_hp": (0, 0, 255),         # 红色
        "player_stamina": (0, 255, 255), # 黄色
    }
    
    for region_key, region in BLOOD_REGION.items():
        x, y, w, h = region["left"], region["top"], region["width"], region["height"]
        color = colors.get(region_key, (255, 255, 255))
        
        # 画矩形
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        
        # 标签
        label = f"{region_key} ({w}x{h})"
        cv2.putText(annotated, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return annotated


def visualize(path=None):
    """可视化血条区域和HSV掩码"""
    frame = load_image(path)
    if frame is None:
        return
    
    # 标注区域
    annotated = draw_regions(frame)
    
    # 为每个区域显示HSV掩码
    detector = BloodDetector()
    
    for region_key in BLOOD_REGION:
        region = BLOOD_REGION[region_key]
        x, y, w, h = region["left"], region["top"], region["width"], region["height"]
        
        # 裁剪
        bar = frame[y:y+h, x:x+w].copy()
        if bar.size == 0:
            print(f"  [WARN] {region_key} 区域为空，可能坐标不正确")
            continue
        
        # HSV掩码
        hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        hsv_range = HSV_RANGES.get(region_key)
        
        if hsv_range:
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
            
            blood_ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])
            
            # 掩码可视化（红色叠加）
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 2] = np.where(mask > 0, 255, mask_colored[:, :, 2])
            
            # 检测结果
            hp = detector.detect_hp(frame, region_key)
            print(f"  {region_key}: HP={hp:.1%} (mask ratio={blood_ratio:.1%})")
        else:
            mask_colored = np.zeros_like(bar)
            hp = detector.detect_hp(frame, region_key)
            print(f"  {region_key}: HP={hp:.1%} (no HSV range, brightness fallback)")
    
    # 显示
    cv2.imshow("Blood Regions", annotated)
    
    # 显示每个区域的特写
    for region_key in BLOOD_REGION:
        region = BLOOD_REGION[region_key]
        x, y, w, h = region["left"], region["top"], region["width"], region["height"]
        bar = frame[y:y+h, x:x+w]
        if bar.size == 0:
            continue
        
        # 放大显示（血条很窄，放大4倍方便观察）
        scale = max(4, 200 // max(h, 1))
        bar_large = cv2.resize(bar, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"{region_key} (x{scale})", bar_large)
    
    print("\n[Visualize] 按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def scan_hsv(path=None):
    """扫描血条区域，自动推荐HSV范围"""
    frame = load_image(path)
    if frame is None:
        return
    
    print("\n[Scan] 扫描血条区域HSV分布...")
    print("=" * 60)
    
    for region_key in BLOOD_REGION:
        region = BLOOD_REGION[region_key]
        x, y, w, h = region["left"], region["top"], region["width"], region["height"]
        bar = frame[y:y+h, x:x+w]
        
        if bar.size == 0:
            print(f"  {region_key}: 区域为空，跳过")
            continue
        
        hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        
        # 分析非背景像素的HSV分布
        # 血条区域通常有背景（暗色）和血量（亮色/彩色）
        # 用亮度(V)阈值分离
        v_channel = hsv[:, :, 2]
        
        # 去掉最暗的30%（背景）和最亮的5%（反光）
        v_sorted = v_channel.flatten()
        v_low = np.percentile(v_sorted, 30)
        v_high = np.percentile(v_sorted, 95)
        
        # 提取中间亮度的像素
        mask_v = (v_channel >= v_low) & (v_channel <= v_high)
        
        if mask_v.sum() == 0:
            print(f"  {region_key}: 无法分离前景，请检查区域坐标")
            continue
        
        h_vals = hsv[:, :, 0][mask_v]
        s_vals = hsv[:, :, 1][mask_v]
        v_vals = hsv[:, :, 2][mask_v]
        
        # 统计H通道分布
        h_hist, _ = np.histogram(h_vals, bins=36, range=(0, 180))
        dominant_bins = np.argsort(h_hist)[-3:][::-1]
        
        print(f"\n  {region_key} (区域: x={x}, y={y}, w={w}, h={h}):")
        print(f"    前景像素: {mask_v.sum()}/{v_channel.size}")
        print(f"    H通道 - min={h_vals.min()}, max={h_vals.max()}, mean={h_vals.mean():.1f}")
        print(f"    S通道 - min={s_vals.min()}, max={s_vals.max()}, mean={s_vals.mean():.1f}")
        print(f"    V通道 - min={v_vals.min()}, max={v_vals.max()}, mean={v_vals.mean():.1f}")
        print(f"    主色调bin: {[f'{b*5}~{(b+1)*5}' for b in dominant_bins]}")
        
        # 推荐HSV范围
        h_min, h_max = h_vals.min(), h_vals.max()
        s_min, s_max = s_vals.min(), s_vals.max()
        v_min_actual, v_max_actual = v_vals.min(), v_vals.max()
        
        # 扩展范围10%以增加鲁棒性
        h_margin = max(5, (h_max - h_min) * 0.1)
        s_margin = max(10, (s_max - s_min) * 0.1)
        v_margin = max(10, (v_max_actual - v_min_actual) * 0.1)
        
        h_lo = max(0, h_min - h_margin)
        h_hi = min(180, h_max + h_margin)
        
        # 检查是否跨越0度（红色）
        if h_min < 15 or h_max > 165:
            # 红色跨越0度，需要两段
            print(f"    ⚠️ 检测到红色色调（H跨越0度）")
            print(f"    推荐HSV范围:")
            print(f'      "lower": [{max(0,int(h_min-h_margin))}, {max(0,int(s_min-s_margin))}, {max(0,int(v_min_actual-v_margin))}],')
            print(f'      "upper": [{min(15,int(h_max+h_margin))}, 255, 255],')
            print(f'      "lower2": [{max(165,int(180-h_margin))}, {max(0,int(s_min-s_margin))}, {max(0,int(v_min_actual-v_margin))}],')
            print(f'      "upper2": [180, 255, 255]')
        else:
            print(f"    推荐HSV范围:")
            print(f'      "lower": [{int(h_lo)}, {max(0,int(s_min-s_margin))}, {max(0,int(v_min_actual-v_margin))}],')
            print(f'      "upper": [{int(h_hi)}, 255, 255]')
        
        # 显示特写
        scale = max(4, 200 // max(h, 1))
        bar_large = cv2.resize(bar, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"{region_key} (x{scale})", bar_large)
    
    print("\n[Scan] 按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_detection(path=None):
    """快速测试当前配置的检测效果"""
    frame = load_image(path)
    if frame is None:
        return
    
    detector = BloodDetector()
    vitals = detector.get_all_vitals(frame)
    
    print("\n[Test] 血量检测结果:")
    print("=" * 40)
    for key, value in vitals.items():
        if "delta" in key:
            sign = "+" if value >= 0 else ""
            print(f"  {key}: {sign}{value:.4f}")
        else:
            bar_len = int(value * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {key}: {value:.1%} |{bar}|")
    
    # 标注到画面上
    annotated = draw_regions(frame)
    
    # 在每个区域旁边标注检测值
    colors = {
        "player_hp": (0, 255, 0),
        "boss_hp": (0, 0, 255),
        "player_stamina": (0, 255, 255),
    }
    
    for region_key in BLOOD_REGION:
        region = BLOOD_REGION[region_key]
        x, y = region["left"], region["top"]
        hp = vitals.get(region_key, 0)
        color = colors.get(region_key, (255, 255, 255))
        text = f"{hp:.1%}"
        cv2.putText(annotated, text, (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # 显示结果
    cv2.imshow("Detection Results", annotated)
    print("\n按任意键关闭...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def tune_interactive(path=None):
    """交互式调整HSV参数（使用OpenCV trackbar）"""
    frame = load_image(path)
    if frame is None:
        return
    
    print("\n[Tune] 交互式HSV调整")
    print("  使用滑块调整参数，按 's' 保存，按 'q' 退出")
    print("  使用数字键 1/2/3 切换 region: 1=player_hp, 2=boss_hp, 3=stamina")
    
    regions = list(BLOOD_REGION.keys())
    current_idx = 0
    
    def nothing(x):
        pass
    
    # 创建窗口
    cv2.namedWindow("HSV Tune", cv2.WINDOW_NORMAL)
    
    # 创建滑块 - 初始值来自当前配置
    region_key = regions[current_idx]
    hsv_range = HSV_RANGES.get(region_key, {})
    
    lower = hsv_range.get("lower", [0, 100, 100])
    upper = hsv_range.get("upper", [10, 255, 255])
    
    cv2.createTrackbar("H_lo", "HSV Tune", lower[0], 180, nothing)
    cv2.createTrackbar("S_lo", "HSV Tune", lower[1], 255, nothing)
    cv2.createTrackbar("V_lo", "HSV Tune", lower[2], 255, nothing)
    cv2.createTrackbar("H_hi", "HSV Tune", upper[0], 180, nothing)
    cv2.createTrackbar("S_hi", "HSV Tune", upper[1], 255, nothing)
    cv2.createTrackbar("V_hi", "HSV Tune", upper[2], 255, nothing)
    
    # 第二区间滑块
    has_lower2 = "lower2" in hsv_range
    lower2 = hsv_range.get("lower2", [170, 100, 100])
    upper2 = hsv_range.get("upper2", [180, 255, 255])
    
    cv2.createTrackbar("H2_lo", "HSV Tune", lower2[0], 180, nothing)
    cv2.createTrackbar("S2_lo", "HSV Tune", lower2[1], 255, nothing)
    cv2.createTrackbar("V2_lo", "HSV Tune", lower2[2], 255, nothing)
    cv2.createTrackbar("H2_hi", "HSV Tune", upper2[0], 180, nothing)
    cv2.createTrackbar("S2_hi", "HSV Tune", upper2[1], 255, nothing)
    cv2.createTrackbar("V2_hi", "HSV Tune", upper2[2], 255, nothing)
    
    cv2.createTrackbar("Range2", "HSV Tune", 1 if has_lower2 else 0, 1, nothing)
    
    saved_config = {}
    
    while True:
        # 读取滑块值
        h_lo = cv2.getTrackbarPos("H_lo", "HSV Tune")
        s_lo = cv2.getTrackbarPos("S_lo", "HSV Tune")
        v_lo = cv2.getTrackbarPos("V_lo", "HSV Tune")
        h_hi = cv2.getTrackbarPos("H_hi", "HSV Tune")
        s_hi = cv2.getTrackbarPos("S_hi", "HSV Tune")
        v_hi = cv2.getTrackbarPos("V_hi", "HSV Tune")
        
        h2_lo = cv2.getTrackbarPos("H2_lo", "HSV Tune")
        s2_lo = cv2.getTrackbarPos("S2_lo", "HSV Tune")
        v2_lo = cv2.getTrackbarPos("V2_lo", "HSV Tune")
        h2_hi = cv2.getTrackbarPos("H2_hi", "HSV Tune")
        s2_hi = cv2.getTrackbarPos("S2_hi", "HSV Tune")
        v2_hi = cv2.getTrackbarPos("V2_hi", "HSV Tune")
        
        use_range2 = cv2.getTrackbarPos("Range2", "HSV Tune") == 1
        
        # 裁剪当前区域
        region = BLOOD_REGION[region_key]
        x, y, w, h = region["left"], region["top"], region["width"], region["height"]
        bar = frame[y:y+h, x:x+w]
        
        if bar.size > 0:
            hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
            
            # 计算掩码
            mask1 = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo]), np.array([h_hi, s_hi, v_hi]))
            
            if use_range2:
                mask2 = cv2.inRange(hsv, np.array([h2_lo, s2_lo, v2_lo]), np.array([h2_hi, s2_hi, v2_hi]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = mask1
            
            ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1]) if mask.size > 0 else 0
            
            # 可视化
            # 上方：原图放大
            scale = max(4, 200 // max(h, 1))
            bar_large = cv2.resize(bar, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            
            # 下方：掩码
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # 用红色标记匹配像素
            result = bar.copy()
            result[mask > 0] = [0, 0, 255]  # 红色标记
            result_large = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            
            # 拼接
            display = np.vstack([bar_large, result_large])
            
            # 标注信息
            info = f"{region_key}: {ratio:.1%} | H:[{h_lo}-{h_hi}] S:[{s_lo}-{s_hi}] V:[{v_lo}-{v_hi}]"
            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if use_range2:
                info2 = f"Range2: H:[{h2_lo}-{h2_hi}] S:[{s2_lo}-{s2_hi}] V:[{v2_lo}-{v2_hi}]"
                cv2.putText(display, info2, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("HSV Tune", display)
        
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存当前参数
            new_range = {
                "lower": [h_lo, s_lo, v_lo],
                "upper": [h_hi, s_hi, v_hi],
            }
            if use_range2:
                new_range["lower2"] = [h2_lo, s2_lo, v2_lo]
                new_range["upper2"] = [h2_hi, s2_hi, v2_hi]
            
            saved_config[region_key] = new_range
            print(f"\n  [Saved] {region_key}:")
            print(f"    {json.dumps(new_range, indent=2)}")
        elif key == ord('1') and len(regions) > 0:
            current_idx = 0
            region_key = regions[0]
            print(f"  [Switch] → {region_key}")
        elif key == ord('2') and len(regions) > 1:
            current_idx = 1
            region_key = regions[1]
            print(f"  [Switch] → {region_key}")
        elif key == ord('3') and len(regions) > 2:
            current_idx = 2
            region_key = regions[2]
            print(f"  [Switch] → {region_key}")
    
    cv2.destroyAllWindows()
    
    if saved_config:
        print("\n[Result] 保存的HSV配置（复制到 config.py 的 HSV_RANGES 中）:")
        print("=" * 60)
        print("HSV_RANGES = {")
        for rk, rv in saved_config.items():
            print(f'    "{rk}": {json.dumps(rv, indent=8)},')
        # 保留未修改的区域
        for rk in BLOOD_REGION:
            if rk not in saved_config:
                print(f'    "{rk}": {json.dumps(HSV_RANGES[rk], indent=8)},')
        print("}")
    else:
        print("\n[Tune] 未保存任何参数")


def tune_region_interactive(path=None):
    """交互式调整血条区域坐标"""
    frame = load_image(path)
    if frame is None:
        return
    
    print("\n[Region Tune] 交互式区域坐标调整")
    print("  拖动矩形调整区域位置和大小")
    print("  按 '1/2/3' 选择区域，按 's' 保存，按 'q' 退出")
    
    regions = list(BLOOD_REGION.keys())
    current_idx = 0
    dragging = False
    start_point = None
    
    # 缩小显示（1920x1080太大）
    scale = 0.5
    display = cv2.resize(frame, None, fx=scale, fy=scale)
    
    region_rects = {}
    for rk, rv in BLOOD_REGION.items():
        region_rects[rk] = {
            "x": int(rv["left"] * scale),
            "y": int(rv["top"] * scale),
            "w": int(rv["width"] * scale),
            "h": int(rv["height"] * scale),
        }
    
    colors = {
        "player_hp": (0, 255, 0),
        "boss_hp": (0, 0, 255),
        "player_stamina": (0, 255, 255),
    }
    
    saved_regions = {}
    
    def draw_all():
        d = display.copy()
        for i, rk in enumerate(regions):
            r = region_rects[rk]
            c = colors.get(rk, (255, 255, 255))
            thickness = 3 if i == current_idx else 1
            cv2.rectangle(d, (r["x"], r["y"]), (r["x"]+r["w"], r["y"]+r["h"]), c, thickness)
            label = f"{i+1}: {rk}"
            cv2.putText(d, label, (r["x"], r["y"]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        return d
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal dragging, start_point, region_rects
        
        rk = regions[current_idx]
        
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            start_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            if start_point:
                # 计算新矩形
                x1, y1 = min(start_point[0], x), min(start_point[1], y)
                x2, y2 = max(start_point[0], x), max(start_point[1], y)
                if x2 - x1 > 5 and y2 - y1 > 5:
                    region_rects[rk] = {"x": x1, "y": y1, "w": x2-x1, "h": y2-y1}
        elif event == cv2.EVENT_MOUSEMOVE and dragging and start_point:
            d = draw_all()
            cv2.rectangle(d, start_point, (x, y), colors.get(rk, (255,255,255)), 2)
            cv2.imshow("Region Tune", d)
    
    cv2.namedWindow("Region Tune", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Region Tune", mouse_callback)
    
    while True:
        d = draw_all()
        cv2.imshow("Region Tune", d)
        
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1') and len(regions) > 0:
            current_idx = 0
            print(f"  → {regions[0]}")
        elif key == ord('2') and len(regions) > 1:
            current_idx = 1
            print(f"  → {regions[1]}")
        elif key == ord('3') and len(regions) > 2:
            current_idx = 2
            print(f"  → {regions[2]}")
        elif key == ord('s'):
            rk = regions[current_idx]
            r = region_rects[rk]
            # 转换回原始坐标
            saved_regions[rk] = {
                "top": int(r["y"] / scale),
                "left": int(r["x"] / scale),
                "width": int(r["w"] / scale),
                "height": int(r["h"] / scale),
            }
            print(f"  [Saved] {rk}: {saved_regions[rk]}")
    
    cv2.destroyAllWindows()
    
    if saved_regions:
        print("\n[Result] 保存的区域配置（复制到 config.py 的 BLOOD_REGION 中）:")
        print("=" * 60)
        print("BLOOD_REGION = {")
        for rk in BLOOD_REGION:
            if rk in saved_regions:
                rv = saved_regions[rk]
            else:
                rv = BLOOD_REGION[rk]
            print(f'    "{rk}": {{')
            print(f'        "top": {rv["top"]},')
            print(f'        "left": {rv["left"]},')
            print(f'        "width": {rv["width"]},')
            print(f'        "height": {rv["height"]},')
            print(f'    }},')
        print("}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    img_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if command == "capture":
        capture_game()
    elif command == "visualize":
        visualize(img_path)
    elif command == "scan":
        scan_hsv(img_path)
    elif command == "test":
        test_detection(img_path)
    elif command == "tune":
        tune_interactive(img_path)
    elif command == "region":
        tune_region_interactive(img_path)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
