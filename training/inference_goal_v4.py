"""
inference_goal_v4.py - v4.1模型专用推理脚本

使用方法：
  cd D:\projects\wukong_ai
  C:\Python\python.exe -u training/inference_goal_v4.py --model checkpoints/goal_bc_v4_final.pt --goal-id 0 --duration 60

注意事项：
  1. 模型输出3个动作：0=forward, 1=right, 2=left
  2. 鼠标dx范围约±86.5，dy约±20.2
  3. 推理时目标方向偏差超过一定阈值才转向
"""

import os, sys, time, argparse, torch, cv2, numpy as np
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.goal_conditioned_bc import GoalConditionedBC
from utils.screen_capture import capture_game_frame
from utils.action_executor import execute_action


def load_model(model_path, num_goals=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoalConditionedBC(num_goals=num_goals).to(device)
    
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    print(f"[Load] Model from {model_path}")
    print(f"[Load] Device: {device}")
    return model, device


def forward_turn(model, frame, goal_id, device):
    """前向推理，返回动作和鼠标偏移"""
    # 预处理
    if frame.shape[:2] != (224, 224):
        frame = cv2.resize(frame, (224, 224))
    
    img = frame.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        action_logits, mouse_pred = model(img, torch.tensor([goal_id]).to(device))
    
    # 动作分类（0=forward, 1=right, 2=left）
    _, action_idx = torch.max(action_logits, dim=1)
    action_idx = action_idx.item()
    
    # 鼠标预测（已归一化到±3范围）
    mdx, mdy = mouse_pred[0].cpu().numpy()
    
    return action_idx, mdx, mdy


ACTION_MAP = {
    0: 'forward',  # forward
    1: 'right',     # right turn
    2: 'left'       # left turn
}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model, device = load_model(args.model, args.num_goals)
    
    # 初始化
    action_executor = execute_action()
    capture = capture_game_frame()
    
    print(f"\n{'='*60}")
    print(f"  Inference Started")
    print(f"  Model: {args.model}")
    print(f"  Goal ID: {args.goal_id}")
    print(f"  Duration: {args.duration}s")
    print(f"  Threshold: {args.turn_threshold}")
    print(f"{'='*60}\n")
    
    # 统计
    stats = {
        'forward': 0, 'right': 0, 'left': 0,
        'total': 0, 'start_time': time.time()
    }
    
    last_action = None
    last_time = time.time()
    
    try:
        while True:
            elapsed = time.time() - stats['start_time']
            if args.duration > 0 and elapsed >= args.duration:
                print("\n[Done] Duration reached")
                break
            
            # 截取游戏画面
            frame = capture()
            if frame is None:
                time.sleep(0.05)
                continue
            
            # 前向推理
            action_idx, mdx, mdy = forward_turn(model, frame, args.goal_id, device)
            
            action_name = ACTION_MAP.get(action_idx, 'unknown')
            stats[action_name] += 1
            stats['total'] += 1
            
            # 鼠标阈值过滤（只有当|mouse|>阈值才真正移动鼠标）
            if abs(mdx) > args.turn_threshold:
                # 执行鼠标移动（mdy是垂直方向，通常设为0）
                move_x = int(mdx)
                move_y = 0  # 水平游戏，垂直鼠标通常为0
                action_executor.move_mouse(move_x, move_y)
            
            # 执行动作
            action_executor.execute(action_name)
            
            # 打印
            if stats['total'] % 30 == 0:
                fps = stats['total'] / elapsed
                print(f"  [{elapsed:.0f}s] FPS: {fps:.1f} | "
                      f"Forward: {stats['forward']} Right: {stats['right']} Left: {stats['left']} | "
                      f"Action: {action_name} Mouse: ({mdx:.1f}, {mdy:.1f})")
            
            # 控制帧率
            time.sleep(0.016)  # ~60fps
    
    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user")
    
    # 统计摘要
    total = stats['total']
    print(f"\n{'='*60}")
    print(f"  Inference Summary")
    print(f"  Total frames: {total}")
    print(f"  Forward: {stats['forward']} ({100*stats['forward']/max(total,1):.1f}%)")
    print(f"  Right: {stats['right']} ({100*stats['right']/max(total,1):.1f}%)")
    print(f"  Left: {stats['left']} ({100*stats['left']/max(total,1):.1f}%)")
    print(f"  Duration: {time.time() - stats['start_time']:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--goal-id", type=int, default=0, help="目标ID (0=进门向右, 1=进门向左)")
    parser.add_argument("--num-goals", type=int, default=2, help="目标数量")
    parser.add_argument("--duration", type=int, default=0, help="运行时间(秒)，0=无限")
    parser.add_argument("--turn-threshold", type=float, default=0.5, help="鼠标移动阈值")
    parser.add_argument("--mouse-scale", type=float, default=1.0, help="鼠标移动缩放")
    args = parser.parse_args()
    
    main(args)