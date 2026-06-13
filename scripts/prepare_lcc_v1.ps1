param(
    [string]$Source = "pathfinding_data_lcc_v1",
    [string]$Prepared = "pathfinding_data_lcc_v1_prepared",
    [string]$Manifest = "dataset_manifests\pathfinding_lcc_v1.json",
    [double]$MouseDxScale = 1.0
)

$ErrorActionPreference = "Stop"
Set-Location "D:\projects\wukong_ai"

& "C:\Python\python.exe" -u "data_tools\prepare_lcc_dataset.py" `
    --source-dir $Source `
    --prepared-dir $Prepared `
    --manifest $Manifest `
    --mouse-dx-scale $MouseDxScale `
    --replace-prepared
