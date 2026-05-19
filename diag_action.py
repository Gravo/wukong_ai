import time, pydirectinput
from env.action_executor import ActionExecutor

executor = ActionExecutor()
print("5秒后连续执行 move_left(6) 3次...")
time.sleep(5)

for i in range(3):
    result = executor.execute(6)  # move_left = A键
    print(f"  第{i+1}次: {result}")
    time.sleep(0.5)

print("\n5秒后连续执行 move_forward(4) 3次...")
time.sleep(5)

for i in range(3):
    result = executor.execute(4)  # move_forward = W键
    print(f"  第{i+1}次: {result}")
    time.sleep(0.5)

print("\n完成！请观察游戏角色是否移动了")
