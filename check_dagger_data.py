"""检查DAgger数据质量"""
import h5py
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python check_dagger_data.py <h5_file>")
    sys.exit(1)

filepath = sys.argv[1]
print(f"\n=== 检查文件: {filepath} ===\n")

with h5py.File(filepath, 'r') as f:
    total_frames = f['frames'].shape[0]
    intervention = f['intervention'][()]
    human_actions = f['human_actions'][()]
    ai_actions = f['ai_actions'][()]
    mouse_dx = f['mouse_dx'][()]
    mouse_dy = f['mouse_dy'][()]
    
    n_interventions = np.sum(intervention)
    
    print(f"📊 总帧数: {total_frames}")
    print(f"⚠️  干预帧数: {n_interventions} ({100*n_interventions/total_frames:.1f}%)")
    
    # AI动作分布
    ai_valid = ai_actions[ai_actions >= 0]
    if len(ai_valid) > 0:
        ai_dist = np.bincount(ai_valid)
        print(f"\n🤖 AI动作分布:")
        action_names = ["idle", "attack", "heavy", "dodge", "forward", "right", "left"]
        for i, count in enumerate(ai_dist):
            if i < len(action_names):
                print(f"  {action_names[i]}: {count} ({100*count/len(ai_valid):.1f}%)")
    
    # 人类动作分布
    human_valid = human_actions[human_actions >= 0]
    if len(human_valid) > 0:
        human_dist = np.bincount(human_valid)
        print(f"\n👨 人类动作分布:")
        for i, count in enumerate(human_dist):
            if i < len(action_names):
                print(f"  {action_names[i]}: {count} ({100*count/len(human_valid):.1f}%)")
    
    # 鼠标数据质量
    print(f"\n🖱️  鼠标数据:")
    print(f"  dx范围: [{np.min(mouse_dx):.3f}, {np.max(mouse_dx):.3f}]")
    print(f"  dy范围: [{np.min(mouse_dy):.3f}, {np.max(mouse_dy):.3f}]")
    print(f"  dx标准差: {np.std(mouse_dx):.3f}")
    print(f"  dy标准差: {np.std(mouse_dy):.3f}")
    
    # 评估
    print(f"\n✅ 质量评估:")
    if n_interventions < 50:
        print(f"  ⚠️  干预帧数偏少（{n_interventions}/建议50+），建议继续采集")
    else:
        print(f"  ✅ 干预帧数充足（{n_interventions}），可以开始训练")
    
    if np.std(mouse_dx) < 0.1 and np.std(mouse_dy) < 0.1:
        print(f"  ⚠️  鼠标数据变化太小，可能未正确记录人工纠正")
    else:
        print(f"  ✅ 鼠标数据有合理变化，记录正常")
    
print(f"\n{'='*50}")
