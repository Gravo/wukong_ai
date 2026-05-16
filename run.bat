"""
run.bat - Windows 一键启动脚本
快速运行 wukong_ai 各种模式
"""
@echo off
chcp 65001 >nul 2>&1
set PYTHON=C:\Python\python.exe
set PROJECT=D:\projects\wukong_ai

echo ============================================
echo   wukong_ai - 黑神话：悟空 AI
echo ============================================
echo.
echo   1. 系统测试 (quick_test)
echo   2. 血量校准 (calibrate)
echo   3. 数据采集 (collect)
echo   4. 行为克隆训练 (bc_train)
echo   5. PPO训练 (train)
echo   6. PPO评估 (eval)
echo.

set /p choice="请选择模式 (1-6): "

if "%choice%"=="1" (
    %PYTHON% %PROJECT%\tools\quick_test.py
) else if "%choice%"=="2" (
    echo.
    echo   calibrate 子命令:
    echo     capture    - 截取游戏画面
    echo     visualize  - 可视化血条区域
    echo     scan       - 自动扫描HSV范围
    echo     test       - 测试当前配置
    echo     tune       - 交互式HSV调整
    echo     region     - 交互式区域调整
    echo.
    set /p subcmd="输入子命令: "
    %PYTHON% %PROJECT%\tools\calibrate_blood.py %subcmd%
) else if "%choice%"=="3" (
    set /p mode="模式 (pathfinding/combat): "
    set /p eps="录制轮数 (默认5): "
    if "%eps%"=="" set eps=5
    %PYTHON% %PROJECT%\training\data_collector.py --mode %mode% --episodes %eps%
) else if "%choice%"=="4" (
    %PYTHON% %PROJECT%\pathfinding\behavior_clone.py
) else if "%choice%"=="5" (
    %PYTHON% %PROJECT%\training\train_combat.py train
) else if "%choice%"=="6" (
    set /p model="模型路径: "
    %PYTHON% %PROJECT%\training\train_combat.py eval --model %model%
) else (
    echo 无效选择
)

echo.
pause
