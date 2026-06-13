param(
    [ValidateSet(
        "KEEP_CENTER",
        "TURN_LEFT_SOON",
        "TURN_RIGHT_SOON",
        "AVOID_LEFT_WALL",
        "AVOID_RIGHT_WALL",
        "ENTER_LEFT_OPENING",
        "ENTER_RIGHT_OPENING",
        "RECOVER_FROM_STUCK"
    )]
    [string]$Command = "KEEP_CENTER",
    [int]$Duration = 6,
    [int]$Fps = 15,
    [int]$Episodes = 1,
    [string]$OutputRoot = "pathfinding_data_lcc_cmd_v1"
)

$ErrorActionPreference = "Stop"
Set-Location "D:\projects\wukong_ai"

$Output = Join-Path $OutputRoot $Command
New-Item -ItemType Directory -Force -Path $Output | Out-Null

& "C:\Python\python.exe" -u "training\data_collector_v3.py" `
    --mode pathfinding `
    --output $Output `
    --fps $Fps `
    --duration $Duration `
    --episodes $Episodes `
    --auto-save 0 `
    --goals "LCC_$Command"
