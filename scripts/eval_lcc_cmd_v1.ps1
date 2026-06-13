param(
    [int]$Duration = 45,
    [int]$CommandId = 0,
    [string]$RunId = "001",
    [string]$Model = "D:\projects\wukong_ai\checkpoints\v63_lcc_cmd_v1_history\goal_bc_v56_history_best_action.pt"
)

$ErrorActionPreference = "Stop"
Set-Location "D:\projects\wukong_ai"

& "C:\Python\python.exe" -u "apps\run_inference_v2.py" `
    --model $Model `
    --policy v56-history `
    --goal-id $CommandId `
    --controller vigem `
    --capture screen `
    --duration $Duration `
    --conf-threshold 0.35 `
    --telemetry "D:\projects\wukong_ai\telemetry\v63_lcc_cmd${CommandId}_eval_$RunId.csv"
