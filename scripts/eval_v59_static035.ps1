param(
    [int]$Duration = 60,
    [string]$RunId = "001"
)

$ErrorActionPreference = "Stop"
Set-Location "D:\projects\wukong_ai"

& "C:\Python\python.exe" -u "apps\run_inference_v2.py" `
    --model "D:\projects\wukong_ai\checkpoints\v59_dagger_start_dx3_repeat5_frozen\goal_bc_v59_dagger_best_action.pt" `
    --policy v56-history `
    --goal-id 1 `
    --controller vigem `
    --capture screen `
    --duration $Duration `
    --conf-threshold 0.35 `
    --telemetry "D:\projects\wukong_ai\telemetry\v59_static035_eval_$RunId.csv"
