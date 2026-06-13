param(
    [string]$Source = "pathfinding_data_lcc_cmd_v1",
    [string]$Prepared = "pathfinding_data_lcc_cmd_v1_prepared",
    [string]$Manifest = "dataset_manifests\pathfinding_lcc_cmd_v1.json",
    [string]$CommandMap = "",
    [string]$DefaultCommand = "",
    [double]$MouseDxScale = 1.0
)

$ErrorActionPreference = "Stop"
Set-Location "D:\projects\wukong_ai"

$ArgsList = @(
    "-u", "data_tools\prepare_command_lcc_dataset.py",
    "--source-dir", $Source,
    "--prepared-dir", $Prepared,
    "--manifest", $Manifest,
    "--mouse-dx-scale", $MouseDxScale,
    "--replace-prepared"
)

if ($CommandMap) {
    $ArgsList += @("--command-map", $CommandMap)
}
if ($DefaultCommand) {
    $ArgsList += @("--default-command", $DefaultCommand)
}

& "C:\Python\python.exe" @ArgsList
