param(
    [int]$Duration = 60,
    [string]$RunId = "001",
    [string]$GoalModel = "D:\projects\wukong_ai\checkpoints\v59_dagger_start_dx3_repeat5_frozen\goal_bc_v59_dagger_best_action.pt",
    [string]$LccModel = "D:\projects\wukong_ai\checkpoints\v62_lcc_v1_history\goal_bc_v56_history_best_action.pt"
)

$ErrorActionPreference = "Stop"
Set-Location "D:\projects\wukong_ai"

& "C:\Python\python.exe" -u "apps\run_inference_v2.py" `
    --model $GoalModel `
    --policy v56-history `
    --goal-id 1 `
    --lcc-model $LccModel `
    --lcc-policy v56-history `
    --lcc-threshold 0.55 `
    --lcc-override-frames 4 `
    --controller vigem `
    --capture screen `
    --duration $Duration `
    --conf-threshold 0.35 `
    --telemetry "D:\projects\wukong_ai\telemetry\v62_goal_with_lcc_v1_eval_$RunId.csv"
