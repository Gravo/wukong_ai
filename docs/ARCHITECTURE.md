# wukong_ai 架构升级说明

本项目现在按“可插拔智能体平台”组织，而不是单脚本实验集合。

## 设计原则

1. **DIP 依赖倒置**
   高层运行逻辑只依赖 `core/ports.py` 中的抽象端口，不直接依赖 `dxcam`、`pydirectinput`、`vigemclient` 或某个具体模型。

2. **Facade**
   `services/game_agent_facade.py` 统一编排：

   ```text
   capture -> frame buffer -> policy -> confidence gate -> controller
   ```

3. **Composition Root**
   `services/agent_factory.py` 是对象装配中心。入口脚本只解析配置，不手动拼依赖。

4. **Pipeline 思维**
   推理、DAgger、评估后续都应复用同一个运行内核，只替换 recorder/evaluator，不复制循环。

5. **实验可追踪**
   `storage/experiment_registry.py` 用 SQLite 记录 run 和 eval result。大文件继续放磁盘，数据库只存元数据。

## 当前模块

```text
core/
  ports.py              抽象端口：CapturePort, PolicyPort, ControllerPort
  runtime_config.py     运行配置
  types.py              Prediction, FrameBatch, AgentStep

capture/
  screen_capture_adapter.py
  pyautogui_capture.py
  factory.py

controllers/
  vigem_controller.py
  pydirect_controller.py
  dry_run_controller.py
  factory.py

policies/
  v55_policy.py
  factory.py

services/
  game_agent_facade.py
  agent_factory.py

storage/
  experiment_registry.py

apps/
  run_inference_v2.py
```

## 推荐运行

ViGEm 控制：

```powershell
C:\Python\python.exe -u apps\run_inference_v2.py ^
  --model "D:\projects\wukong_ai\checkpoints\goal_bc_v55_best_acc_a.pt" ^
  --goal-id 1 ^
  --controller vigem ^
  --capture screen ^
  --duration 60 ^
  --registry "D:\projects\wukong_ai\runs\agent_registry.sqlite"
```

无输入 dry-run：

```powershell
C:\Python\python.exe -u apps\run_inference_v2.py ^
  --model "D:\projects\wukong_ai\checkpoints\goal_bc_v55_best_acc_a.pt" ^
  --goal-id 1 ^
  --controller dry-run ^
  --duration 20
```

旧入口仍可用：

```powershell
C:\Python\python.exe -u training\inference_goal_v55.py ^
  --model "D:\projects\wukong_ai\checkpoints\goal_bc_v55_best_acc_a.pt" ^
  --goal-id 1 ^
  --controller vigem ^
  --duration 60
```

## 下一步扩展

1. 新增 `policies/v56_lstm_policy.py`，在 `policies/factory.py` 注册。
2. 新增 DAgger recorder，挂到 `GameAgentFacade` 的运行循环旁路。
3. 新增 evaluator，将 success/intervention/stuck 等指标写入 SQLite。
4. 新增 dataset registry，把 h5 数据集版本、来源、平衡策略记录下来。
