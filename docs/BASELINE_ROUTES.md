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

## Route E: v6.2 LCC Foundation Baseline

Goal is local corridor keeping, independent of Tiger Vanguard or any boss route.

Data:

```powershell
.\scripts\collect_v57_corrections.ps1 -Duration 10 -Output pathfinding_data_lcc_v1
.\scripts\prepare_lcc_v1.ps1
```

Train:

```powershell
.\scripts\train_lcc_v1.ps1
```

Evaluate LCC alone:

```powershell
.\scripts\eval_lcc_v1.ps1 -RunId 001
```

Evaluate goal navigation with LCC assist:

```powershell
.\scripts\eval_goal_with_lcc_v1.ps1 -RunId 001
```

The first LCC-assisted fusion is conservative: the goal policy remains default,
and the LCC policy only overrides when it predicts a confident local turn.

## Route F: v6.3 Command-Conditioned LCC

Goal is to keep the lower LCC layer small, but make it steerable by a discrete
local intent. This is the first baseline toward language/control fusion:

- `KEEP_CENTER`
- `TURN_LEFT_SOON`
- `TURN_RIGHT_SOON`
- `AVOID_LEFT_WALL`
- `AVOID_RIGHT_WALL`
- `ENTER_LEFT_OPENING`
- `ENTER_RIGHT_OPENING`
- `RECOVER_FROM_STUCK`

Data is organized by command directory, and the prepare script writes the
command id into both `command_ids` and `goal_ids`. Reusing `goal_ids` is
intentional for this baseline: the existing goal embedding becomes a command
embedding without changing the model yet.

Collect:

```powershell
.\scripts\collect_lcc_command.ps1 -Command KEEP_CENTER -Duration 6
.\scripts\collect_lcc_command.ps1 -Command AVOID_RIGHT_WALL -Duration 4
.\scripts\collect_lcc_command.ps1 -Command AVOID_LEFT_WALL -Duration 4
```

Prepare:

```powershell
.\scripts\prepare_lcc_cmd_v1.ps1
```

Train:

```powershell
.\scripts\train_lcc_cmd_v1.ps1
```

Evaluate Command-LCC alone:

```powershell
.\scripts\eval_lcc_cmd_v1.ps1 -CommandId 0 -RunId keep_001
.\scripts\eval_lcc_cmd_v1.ps1 -CommandId 4 -RunId avoid_right_001
```

Evaluate goal navigation with Command-LCC assist:

```powershell
.\scripts\eval_goal_with_lcc_cmd_v1.ps1 -CommandId 0 -RunId keep_001
.\scripts\eval_goal_with_lcc_cmd_v1.ps1 -CommandId 4 -RunId avoid_right_001
```

This is not yet a language model. It is the control interface that a small
language or command parser can drive later: text -> command id -> LCC behavior.

## Comparison Checklist

For every run, record:

- Did it leave the start scene?
- Did it pass the known stuck point?
- Did it enter the wrong route?
- Did it reach Tiger Vanguard route?
- Telemetry: execution rate, forward/turn ratio, bucket distribution, skipped frames.
- For Command-LCC: command id, scene, initial pose, and whether the command
  changed behavior compared with `KEEP_CENTER`.

Primary telemetry commands:

```powershell
C:\Python\python.exe -u tools\analyze_telemetry.py telemetry\<file>.csv
```
