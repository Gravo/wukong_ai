#!C:\Python\python.exe
#!/usr/bin/env python3
"""v5.6 推理脚本 - ViGEmBus 手柄控制版"""
import argparse,time,numpy as np,torch,cv2,pyautogui,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
from training.goal_conditioned_bc_v55_optimized import GoalConditionedBC_v55
import vgamepad

BUCKET_STICK={0:-0.80,1:-0.50,2:-0.25,3:0.00,4:+0.25,5:+0.50,6:+0.80}
BUCKET_ACT={0:"turn_left",1:"turn_left",2:"forward",3:"forward",4:"forward",5:"turn_right",6:"turn_right"}

class GP:
    def __init__(self):
        self.pad = vgamepad.VX360Gamepad()
        self.MAX_DEF = 0.80
        self.DEAD_ZONE = 0.05
        # prime: 发送非零偏转，激活游戏手柄模式
        self.pad.right_joystick_float(0.1, 0.0)
        self.pad.update()
        time.sleep(0.1)
        self.release()
        print("[OK] vgamepad ready")

    def _clamp(self, val):
        return max(-self.MAX_DEF, min(self.MAX_DEF, val))

    def _deadzone(self, val):
        if abs(val) < self.DEAD_ZONE:
            return 0.0
        return val

    def release(self):
        self.pad.right_joystick_float(0.0, 0.0)
        self.pad.left_joystick_float(0.0, 0.0)
        self.pad.update()

    def set_sticks(self, lx, ly, rx, ry=0.0):
        lx = self._deadzone(self._clamp(lx))
        ly = self._deadzone(self._clamp(ly))
        rx = self._deadzone(self._clamp(rx))
        ry = self._deadzone(self._clamp(ry))
        self.pad.left_joystick_float(float(lx), float(ly))
        self.pad.right_joystick_float(float(rx), float(ry))
        self.pad.update()

class Inf:
    def __init__(self, model_path, goal=0, dev="cuda:0", conf=0.5):
        self.dev = torch.device(dev if torch.cuda.is_available() else "cpu")
        self.conf = conf
        self.goal = goal
        print("[加载] " + model_path)
        from training.goal_conditioned_bc_v55_optimized import GoalConditionedBC_v55
        self.model = GoalConditionedBC_v55(num_goals=2, freeze_backbone=False)
        ckpt = torch.load(model_path, map_location=self.dev)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            print("[加载] epoch=" + str(ckpt.get("epoch","?")))
        else:
            self.model.load_state_dict(ckpt)
        self.model.to(self.dev).eval()
        print("[加载] device=" + str(self.dev))
        self.gp = GP()
        self.fb = []
        self.last_sx = 0.0

    def prep(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rsz = cv2.resize(rgb, (224, 224))
        norm = rsz.astype(np.float32) / 255.0
        return np.transpose(norm, (2, 0, 1))

    def predict(self, frames):
        t = torch.from_numpy(np.concatenate([self.prep(f) for f in frames], axis=0)).unsqueeze(0).to(self.dev)
        g = torch.tensor([self.goal], dtype=torch.long).to(self.dev)
        with torch.no_grad():
            al, ml = self.model(t, g)
            ap = torch.softmax(al, dim=-1)
            ai, ac = torch.max(ap, dim=-1)
            mp = torch.softmax(ml, dim=-1)
            mb, mc = torch.max(mp, dim=-1)
        return ai.item(), ac.item(), mb.item(), mc.item()

    def execute(self, bucket):
        action = BUCKET_ACT.get(bucket, "forward")
        tsx = BUCKET_STICK.get(bucket, 0.0)
        sx = 0.7 * tsx + 0.3 * self.last_sx
        self.last_sx = sx
        if action == "forward":
            self.gp.set_sticks(0.0, 1.0, sx, 0.0)
        elif action == "turn_left":
            self.gp.set_sticks(-0.3, 0.5, sx, 0.0)
        else:
            self.gp.set_sticks(0.3, 0.5, sx, 0.0)
        return action

    def run(self, dur=60):
        print("\n[推理] v5.6 dur=" + str(dur) + "s Ctrl+C停止\n")
        t0 = time.time()
        fc = 0
        ac = 0
        try:
            while time.time() - t0 < dur:
                scr = pyautogui.screenshot()
                scr = np.array(scr)
                scr = cv2.cvtColor(scr, cv2.COLOR_RGB2BGR)
                self.fb.append(scr)
                if len(self.fb) > 4:
                    self.fb.pop(0)
                if len(self.fb) == 4:
                    ai, ac_, mb, mc = self.predict(self.fb)
                    if mc >= self.conf:
                        action = self.execute(mb)
                        ac += 1
                        if fc % 20 == 0:
                            sx = BUCKET_STICK.get(mb, 0)
                            print("[推理] %d: %s(b=%d sx=%+.2f c=%.2f)" % (fc, action, mb, sx, mc))
                fc += 1
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n[中断]")
        finally:
            self.gp.release()
            print("[完成] %d actions\n" % ac)
        return ac

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=r"D:\projects\wukong_ai\checkpoints\goal_bc_v55_best_acc_a.pt")
    p.add_argument("--goal-id", type=int, default=0)
    p.add_argument("--duration", type=int, default=60)
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda:0")
    a = p.parse_args()
    Inf(a.model, a.goal_id, a.device, a.conf_threshold).run(a.duration)
