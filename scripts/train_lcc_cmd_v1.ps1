param(
    [string]$Manifest = "dataset_manifests\pathfinding_lcc_cmd_v1.json",
    [string]$Output = "checkpoints\v63_lcc_cmd_v1_history",
    [int]$Epochs = 12,
    [int]$BatchSize = 64
)

$ErrorActionPreference = "Stop"
Set-Location "D:\projects\wukong_ai"

& "C:\Python\python.exe" -u "training\train_v56_history_stack_clean.py" `
    --manifest $Manifest `
    --output-dir $Output `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --device cuda:0 `
    --num-goals 8 `
    --freeze-backbone `
    --progress epoch
