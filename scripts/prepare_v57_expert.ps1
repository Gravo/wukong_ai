param(
    [string]$ExpertDir = "pathfinding_data_v57_expert",
    [string]$PreparedDir = "pathfinding_data_v57_expert_goal1",
    [string]$Manifest = "dataset_manifests\pathfinding_v57_expert_augmented_goal1.json",
    [int]$GoalId = 1
)

$ErrorActionPreference = "Stop"

Set-Location "D:\projects\wukong_ai"

& "C:\Python\python.exe" -u "data_tools\prepare_v57_expert.py" `
    --expert-dir $ExpertDir `
    --prepared-dir $PreparedDir `
    --output-manifest $Manifest `
    --target-goal-id $GoalId `
    --min-frames 250 `
    --replace-prepared
