# Baseline Routes

This project now tracks several navigation baselines with the same telemetry
format so closed-loop behavior can be compared instead of guessed.

## Route A: Static Goal BC

- Model: `checkpoints/v59_dagger_start_dx3_repeat5_frozen/goal_bc_v59_dagger_best_action.pt`
- Policy: `v56-history`
- Runtime: fixed confidence threshold.
- Script: `scripts/eval_v59_static035.ps1`
- Purpose: measure the best simple threshold baseline after DAgger data.
- Cost: no extra training, one game run.

## Route B: BLUE-Style Rule Gate

- Model: same as Route A.
- Policy: `v56-history`
- Runtime: `--gate rule`
- Script: `scripts/eval_v60_rule_gate.ps1`
- Purpose: keep normal behavior conservative while allowing weak turn signals
  in recovery states.
- Cost: no extra training, one game run.

The first rule gate is intentionally small:

- normal threshold: `0.50`
- weak turn threshold: `0.35`
- if the model predicts left/right but mouse bucket is straight, convert it to
  an executable turn dx.

This mirrors BLUE's main idea at runtime: open a more complex/corrective branch
only at likely key states, instead of paying that cost every frame.

## Route C: DAgger Corrections

- Data: short failure-state clips.
- Training: `training/train_v59_dagger_history.py`
- Purpose: force failure states into train and repeat them so they are not
  diluted by normal forward data.
- Cost: 10-30 short clips plus one short frozen-backbone training run.

## Route D: Roaming Foundation Skill

Goal is not "go to Tiger Vanguard"; it is local survival/navigation:

- avoid walls
- keep moving through traversable space
- recover from bad pose
- prefer open corridors

Expected cost:

- Data: tens of minutes across several scenes.
- Model: ResNet18/34 + history stack or GRU.
- Runtime: rule gate first, learned gate later.

This should become the base skill that a goal-conditioned route policy builds on.

## Comparison Checklist

For every run, record:

- Did it leave the start scene?
- Did it pass the known stuck point?
- Did it enter the wrong route?
- Did it reach Tiger Vanguard route?
- Telemetry: execution rate, forward/turn ratio, bucket distribution, skipped frames.

Primary telemetry commands:

```powershell
C:\Python\python.exe -u tools\analyze_telemetry.py telemetry\<file>.csv
```
