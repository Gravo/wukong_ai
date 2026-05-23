# record_goal2.ps1 - 录制 Goal 2 (BOSS门/关卡入口)
# 双击运行，或右键"使用 PowerShell 运行"

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Goal 2 录制脚本 (BOSS门/关卡入口)" -ForegroundColor Yellow
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python
Write-Host "[1/4] 检查 Python..." -ForegroundColor Green
try {
    $pythonVersion = & C:\Python\python.exe --version 2>&1
    Write-Host "  ✅ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Python 未找到！请检查 C:\Python\python.exe" -ForegroundColor Red
    Read-Host "按回车退出"
    exit 1
}

# 检查 data_collector_v3.py
Write-Host "[2/4] 检查采集器..." -ForegroundColor Green
$collectorPath = "D:\projects\wukong_ai\training\data_collector_v3.py"
if (-not (Test-Path $collectorPath)) {
    Write-Host "  ❌ 未找到: $collectorPath" -ForegroundColor Red
    Read-Host "按回车退出"
    exit 1
}
Write-Host "  ✅ 找到采集器" -ForegroundColor Green

# 提示操作步骤
Write-Host "[3/4] 准备录制..." -ForegroundColor Green
Write-Host ""
Write-Host "⚠️  重要提示：" -ForegroundColor Yellow
Write-Host "  1. 游戏加载到 存档点B (起始点)"
Write-Host "  2. 在起始点 故意左右转镜头 (让 mouse_dx 有变化)"
Write-Host "  3. 按 G 键，输入 2 (Goal 2 = BOSS门/关卡入口)"
Write-Host "  4. 走到 BOSS门/关卡入口"
Write-Host "  5. 按 ESC 保存"
Write-Host ""
Read-Host "按回车开始录制..."

# 启动采集器
Write-Host "[4/4] 启动采集器..." -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

cd D:\projects\wukong_ai
C:\Python\python.exe -u training\data_collector_v3.py

# 录制完成
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  录制完成！" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步：" -ForegroundColor Yellow
Write-Host "  1. 运行 verify_latest.bat 检查数据质量"
Write-Host "  2. 如果合格，继续录制下一个文件"
Write-Host "  3. 如果不合格，删除并重录"
Write-Host ""

Read-Host "按回车退出"
