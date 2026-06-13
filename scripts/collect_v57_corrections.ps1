param(
    [int]$Duration = 8,
    [int]$Fps = 15,
    [int]$Episodes = 1,
    [string]$Output = "pathfinding_data_v57_corrections"
)

$ErrorActionPreference = "Stop"

Set-Location "D:\projects\wukong_ai"

& "C:\Python\python.exe" -u "training\data_collector_v3.py" `
    --mode pathfinding `
    --output $Output `
    --fps $Fps `
    --duration $Duration `
    --episodes $Episodes `
    --auto-save 0 `
    --goals "虎先锋纠偏"
