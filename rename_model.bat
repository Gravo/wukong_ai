@echo off
cd /d "D:\projects\wukong_ai\checkpoints"
ren "goal_bc_v53_best.pt" "goal_bc_v54_best.pt"
echo File renamed successfully!
dir *v54* 2>nul || echo No v54 files found
pause