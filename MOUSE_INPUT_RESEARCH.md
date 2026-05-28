# 鼠标输入问题 - 紧急研究任务

**日期**: 2026-05-27  
**优先级**: 🔴 **最高** (阻塞整个项目)  
**目标**: 让 AI 控制的鼠标能被游戏识别

---

## 🎯 核心问题

**游戏使用 Raw Input API 读取鼠标**：
- ✅ Windows 光标能移动 (`SendInput` 有效)
- ❌ 游戏内无响应（游戏读硬件，不读 Windows 光标）
- ❌ `pydirectinput` 无效（同样走 Windows 光标）
- ⚠️ 游戏设置中**无 Raw Input 开关**（现代游戏强制开启）

**结果**: 推理测试无法进行，整个项目阻塞。

---

## 🔬 解决方案（按优先级排序）

### 方案1: 研究 Turing 项目（推荐，1-2小时）✅

**项目**: [Turing-Project/Black-Myth-Wukong-AI](https://github.com/Turing-Project/Black-Myth-Wukong-AI)  
**Stars**: 392  
**License**: MIT

**他们肯定解决了鼠标控制问题！**

**研究步骤**:
1. **克隆项目**:
   ```powershell
   cd D:\projects\wukong_ai
   git clone https://github.com/Turing-Project/Black-Myth-Wukong-AI.git
   ```

2. **找鼠标控制代码**:
   - 搜索关键词：`mouse`, `input`, `control`, `SendInput`, `Raw Input`
   - 可能的文件名：`mouse_controller.py`, `input_emulator.py`, `game_input.py`

3. **分析他们的方案**:
   - 如果用 `SendInput` → 可能游戏有漏洞，或他们找到了特殊配置
   - 如果用 `Raw Input Hook` → 需要 C++ DLL 注入
   - 如果用 `视觉方案` → 不依赖鼠标事件，用 OpenCV 识别小地图

4. **复现他们的方案**:
   - 如果代码清晰 → 直接复制过来
   - 如果需要编译 → 写编译脚本
   - 如果依赖复杂 → 提取核心逻辑

**预期时间**: 1-2 小时  
**成功率**: 🟢 高（392 stars 项目，肯定能跑通）

---

### 方案2: Hook Raw Input（高级，2-3小时）⚠️

**原理**: 用 C++ DLL 注入游戏进程，模拟 Raw Input 消息。

**技术细节**:
- Windows 消息：`WM_INPUT`
- 数据结构：`RAWINPUT` struct
- 注入方法：`SetWindowsHookEx` 或 DLL injection

**实现步骤**:
1. **写 C++ DLL** (`raw_input_hook.dll`):
   - Hook `GetRawInputData` 函数
   - 模拟鼠标移动数据（`RAWINPUT` struct）
   - 注入到游戏进程

2. **Python 调用 DLL**:
   - 用 `ctypes` 加载 DLL
   - 调用 `simulate_mouse_move(dx, dy)`

3. **测试**:
   - 运行游戏
   - 调用 `simulate_mouse_move(100, 0)`
   - 看游戏是否响应

**参考代码**:
- [Raw Input Hook Example](https://github.com/nefarius/RawInput.Sharp)
- [DLL Injection Tutorial](https://www.ired.team/offensive-security/code-injection-process-injection)

**预期时间**: 2-3 小时（如果不熟悉 C++，可能 1-2 天）  
**成功率**: 🟡 中（需要 C++ 和 Windows API 知识）

---

### 方案3: 驱动级模拟（高级，1-2天）⚠️

**原理**: 用 Interception 驱动拦截键盘鼠标事件。

**工具**: [Interception](https://github.com/oblitum/Interception)

**实现步骤**:
1. **安装 Interception 驱动**:
   - 下载 `interception.dll`
   - 安装驱动（`install-interception.exe`）

2. **写 Python 调用**:
   - 用 `ctypes` 调用 `interception.dll`
   - 模拟鼠标移动

3. **测试**:
   - 运行游戏
   - 调用模拟函数
   - 看游戏是否响应

**优点**:
- ✅ 驱动级拦截，**所有游戏都能用**（包括反作弊）
- ✅ 不需要注入 DLL

**缺点**:
- ⚠️ 需要安装驱动（有风险）
- ⚠️ 可能被反作弊检测（黑神话有反作弊吗？）

**预期时间**: 1-2 天  
**成功率**: 🟡 中（驱动安装可能失败）

---

### 方案4: 视觉方案（备选，1天）🔵

**原理**: 不用鼠标事件，用 **OpenCV 识别小地图方向**。

**实现步骤**:
1. **识别小地图**:
   - 截图 → OpenCV 模板匹配
   - 找到小地图位置

2. **识别人物方向**:
   - 小地图中心 = 人物位置
   - 小地图箭头 = 人物朝向
   - 用 OpenCV 计算箭头角度

3. **控制逻辑**:
   - 如果朝向 ≠ 目标方向 → 模拟鼠标移动（用 SendInput）
   - 如果朝向 = 目标方向 → 按住 W 前进

**优点**:
- ✅ 不依赖鼠标事件（绕过 Raw Input 问题）
- ✅ 更鲁棒（不受鼠标灵敏度影响）

**缺点**:
- ⚠️ 需要 OpenCV 知识
- ⚠️ 小地图识别可能不准（不同地图、不同光照）

**预期时间**: 1 天  
**成功率**: 🟢 高（OpenCV 成熟）

---

### 方案5: 游戏内存读取（高级，2-3天）🔴

**原理**: 读游戏内存，获取坐标和方向数据。

**需要逆向工程**:
- 用 Cheat Engine 找坐标地址
- 用 `ReadProcessMemory` API 读内存
- 直接计算方向，不需要鼠标

**优点**:
- ✅ 最精确（直接读游戏数据）
- ✅ 不依赖鼠标事件

**缺点**:
- ⚠️ 需要逆向工程知识
- ⚠️ 可能被反作弊检测
- ⚠️ 游戏更新后地址会变

**预期时间**: 2-3 天  
**成功率**: 🔴 低（逆向工程难度大）

---

## 📋 推荐执行顺序

### 第一步：研究 Turing 项目（明天早上，1-2小时）✅

**命令**:
```powershell
cd D:\projects\wukong_ai
git clone https://github.com/Turing-Project/Black-Myth-Wukong-AI.git
cd Black-Myth-Wukong-AI
```

**找鼠标控制代码**:
```powershell
# 搜索关键词
Select-String -Path *.py -Pattern "mouse|input|SendInput|Raw.?Input"

# 或者手动看文件列表
Get-ChildItem -Recurse -Filter *.py | Select-String -Pattern "mouse"
```

**预期结果**:
- ✅ 找到他们的鼠标控制代码
- ✅ 复现他们的方案
- ✅ 问题解决

### 第二步：如果 Turing 项目不行，Hook Raw Input（2-3小时）⚠️

**写 C++ DLL**:
- 参考 [Raw Input Hook Example](https://github.com/nefarius/RawInput.Sharp)
- 编译 DLL

**Python 调用**:
```python
import ctypes
dll = ctypes.CDLL('raw_input_hook.dll')
dll.simulate_mouse_move(100, 0)
```

### 第三步：如果还不行，视觉方案（1天）🔵

**实现 OpenCV 小地图识别**:
- 模板匹配找小地图
- 计算箭头角度
- 控制逻辑

---

## 📊 成功率评估

| 方案 | 成功率 | 时间 | 难度 |
|------|--------|------|------|
| 研究 Turing 项目 | 🟢 80% | 1-2h | 🟢 低 |
| Hook Raw Input | 🟡 50% | 2-3h | 🔴 高 |
| 驱动级模拟 | 🟡 60% | 1-2d | 🔴 高 |
| 视觉方案 | 🟢 90% | 1d | 🟡 中 |
| 游戏内存读取 | 🔴 30% | 2-3d | 🔴 高 |

**推荐**: 先试 **方案1**（研究 Turing 项目），如果不行，用 **方案4**（视觉方案）。

---

## 🎯 明天早上第一时间做

**00:30 之前（今晚）**:
- ✅ 整理好所有文档（`PROGRESS_REPORT.md`, `RETRAIN_PLAN_V53.md`, `MOUSE_INPUT_RESEARCH.md`）
- ✅ 提交 GitHub
- ✅ 去休息

**08:00 之后（明天早上）**:
1. ⏳ **克隆 Turing 项目**（第1优先级）
2. ⏳ **分析他们的鼠标控制代码**
3. ⏳ **复现他们的方案**
4. ⏳ **如果成功，立刻测试推理**
5. ⏳ **如果失败，启动方案4（视觉方案）**

---

## 📞 需要帮助？

**如果遇到问题**:
- 查看 `PROGRESS_REPORT.md` (Known Issues 章节)
- 创建 GitHub Issue（明天手动创建）
- 联系我（Gravo）

---

**最后更新**: 2026-05-27 00:25  
**维护者**: Gao Wei (Gravo)  
**License**: MIT
