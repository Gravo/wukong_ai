param(
    [int]$Duration = 60,
    [string]$RunId = "001",
    [double]$RecoveryThreshold = 0.35,
    [int]$TurnDx = 150
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
    --conf-threshold 0.50 `
    --gate rule `
    --recovery-threshold $RecoveryThreshold `
    --gate-turn-dx $TurnDx `
    --telemetry "D:\projects\wukong_ai\telemetry\v60_rule_gate_eval_$RunId.csv"
