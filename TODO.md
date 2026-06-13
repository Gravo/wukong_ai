# TODO

## Immediate

- [ ] Run `scripts/eval_v61_rule_gate_smooth.ps1` and compare with v59 static.
- [ ] Analyze the v61 telemetry with `tools/analyze_telemetry.py`.
- [ ] Decide whether v61 should become the default runtime gate.

## LCC v1 Dataset

- [ ] Create `pathfinding_data_lcc_v1`.
- [ ] Record 30-50 clips, 5-10 seconds each.
- [ ] Cover centered corridor, left wall correction, right wall correction,
      wall-touch recovery, narrow entrance alignment, bends, and forks.
- [ ] Reject clips with zero mouse activity, long idle, combat, or hard stuck.
- [ ] Build an LCC-specific manifest.
- [ ] Add a short LCC data quality report.
- [ ] Run `scripts/prepare_lcc_v1.ps1` and inspect the manifest summary.

## LCC v1 Model

- [ ] Train a dedicated v56-history LCC baseline.
- [ ] Run `scripts/train_lcc_v1.ps1`.
- [ ] Run `scripts/eval_lcc_v1.ps1` for LCC-only closed-loop behavior.
- [ ] Run `scripts/eval_goal_with_lcc_v1.ps1` for goal + LCC fusion.
- [ ] Compare LCC-only, v59 static, and v61 rule gate on the same local scenes.
- [ ] Add telemetry fields for LCC override decisions if fused with goal policy.
- [ ] Define closed-loop LCC metrics: seconds before stuck, switch count,
      forward/turn balance, and bucket collapse rate.

## Command-Conditioned LCC v1

- [ ] Create `pathfinding_data_lcc_cmd_v1` with one subdirectory per command.
- [ ] Start with three commands: `KEEP_CENTER`, `AVOID_LEFT_WALL`,
      `AVOID_RIGHT_WALL`.
- [ ] Record 10-20 clips per command, 3-6 seconds each, from matched poses
      when possible.
- [ ] Run `scripts/prepare_lcc_cmd_v1.ps1` and inspect command balance.
- [ ] Train `scripts/train_lcc_cmd_v1.ps1`.
- [ ] Compare `KEEP_CENTER` vs wall-avoid commands in the same local scene.
- [ ] Evaluate route policy + command LCC assist with
      `scripts/eval_goal_with_lcc_cmd_v1.ps1`.
- [ ] Decide whether the next input should remain discrete command id or become
      a tiny text encoder feeding the command embedding.

## DAgger Loop

- [ ] Continue forced-train DAgger only for states the model actually fails in.
- [ ] Keep correction data separate from ordinary route data.
- [ ] Repeat/weight extra correction samples instead of relying on normal split.
- [ ] Maintain one manifest per DAgger round.

## Architecture

- [ ] Add a policy fusion layer for goal policy + LCC policy.
- [ ] Promote rule gate parameters to named runtime presets.
- [ ] Add timestamped telemetry names to avoid overwriting repeated runs.
- [ ] Add a learned gate baseline after enough telemetry has been collected.

## Research

- [ ] Revisit temporal GRU after collecting LCC v1 data.
- [ ] Test continuous steering output after discrete LCC baseline is stable.
- [ ] Evaluate no-target roaming as a separate task from boss navigation.
