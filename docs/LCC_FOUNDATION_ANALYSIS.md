# LCC Foundation Model Analysis

## Summary

The current navigation stack has proven that goal-conditioned behavior cloning
can move the character and that DAgger-style short corrections can fix specific
closed-loop failures. It has not yet learned a general local corridor keeping
skill. The next foundation layer should be a small game-LCC module: keep moving
through traversable space, avoid walls, recover from bad pose, and stabilize
local steering before any boss-specific route policy acts.

In driving terms:

- LCC keeps a vehicle centered in a known lane.
- Our game-LCC should keep the character centered in a locally traversable
  corridor, even when there is no explicit lane marker.
- Goal navigation should choose which corridor or exit to take.
- LCC should make that choice physically stable.

The current target "go to Tiger Vanguard" is too short and too narrow to be the
foundation task. It is useful as a closed-loop benchmark, but not enough to
teach reusable navigation.

## Current Stack

The committed runtime is now split into clear ports:

- capture adapters produce BGR frames.
- policy adapters produce `forward / turn_left / turn_right` plus a mouse
  bucket.
- controller adapters translate predictions to keyboard/mouse or ViGEm.
- the facade coordinates capture, policy, controller, telemetry, and optional
  rule gate.

Important baselines:

- `v56-history`: 4 historical frames `[t-7, t-3, t-1, t]` to predict action at
  `t`.
- `v58-temporal`: GRU temporal baseline. Cached-feature version trains quickly,
  but underperforms the CNN history baseline on the current data.
- `v59-dagger`: forced-train, repeated correction data for the known start
  failure state.
- `v60/v61 rule gate`: BLUE-inspired runtime gate that changes execution
  behavior without changing model weights.

## What The Current Model Can Do

The current models can learn:

- move forward in familiar route scenes.
- imitate short human expert routes.
- react to specific corrected failure states after DAgger.
- expose weak turn intent when confidence threshold is lowered.
- pass the original start stuck point after correction data.

The strongest evidence is closed-loop telemetry:

- static `0.35` threshold executes reliably and passes the initial stuck point.
- rule gate can expose turn decisions, but v6.0 over-triggered turns because
  forward was blocked by the higher main threshold.

## What It Cannot Reliably Do Yet

The current models do not yet have a reusable local driving prior:

- they can collapse to `forward + bucket=3` in unseen poses.
- they do not robustly infer "I am drifting toward a wall".
- they do not generalize well across small initial angle changes.
- they do not consistently choose smooth steering amplitude.
- they rely on manual confidence threshold tuning.
- the mouse bucket head is weaker than the action head in closed-loop use.

This means the current system is route-following plus patch corrections, not
yet local navigation.

## Dataset Assessment

Current data is useful but unbalanced for LCC:

| Dataset | Useful For | Limitation |
|---|---|---|
| curated balanced data | general route BC, turn labels | not intentionally organized as LCC states |
| v57 expert route | Tiger Vanguard direction | about 90% forward, encourages route memorization |
| v57 corrections | first DAgger correction | small sample count |
| v59 start corrections | known start stuck point | one/few local failure modes only |

Current data is enough to keep improving the Tiger Vanguard benchmark. It is
not enough for a foundation LCC model because LCC needs broad local pose
coverage:

- centered in corridor
- near left boundary
- near right boundary
- slightly stuck
- narrow entrance alignment
- left bend
- right bend
- fork entry
- recovery after oversteer

## Proposed LCC Task Definition

The LCC foundation should ignore boss identity and focus on local traversability.

Inputs:

- recent frames, initially same `v56-history` format.
- optional short telemetry state: recent actions, recent confidence, recent
  bucket sequence.

Outputs for baseline:

- `forward`
- `turn_left`
- `turn_right`
- `mouse_bucket`

Later output upgrade:

- continuous `right_stick_x`
- optional `forward_intensity`
- optional `recover` mode

The first LCC dataset should not use goal IDs as route targets. It can use a
single local goal id such as `goal_id=0`, or no goal embedding for a dedicated
LCC model.

## Architecture Options

### Option A: Rule Gate LCC

This is the cheapest baseline.

Use the current goal policy, but add runtime rules:

- weak turn signals are allowed at a lower threshold.
- weak forward is also allowed so recovery does not become turn-only.
- when turn action predicts bucket 3, inject a small executable turn dx.
- hold turn direction for a few frames to reduce left/right flicker.

Pros:

- no training required.
- immediately testable.
- transparent telemetry.

Cons:

- rules are brittle.
- does not learn visual wall/corridor features.

### Option B: BC LCC Policy

Train a dedicated LCC policy on local clips.

Model:

- start with `v56-history` CNN stack.
- no boss-specific goal embedding, or a fixed local goal id.

Pros:

- simple and comparable to existing baselines.
- can reuse training and inference infrastructure.

Cons:

- short temporal memory only.
- may still need DAgger to handle closed-loop drift.

### Option C: Temporal LCC Policy

Use GRU over 16-24 frames.

Pros:

- can learn "drifting toward wall" and "stuck over time".
- closer to local driving behavior.

Cons:

- full ResNet+GRU is expensive on RTX 2060.
- cached-feature GRU trained fast but underperformed with current data.
- likely needs a better LCC dataset before it becomes useful.

### Option D: Continuous Control LCC

Predict continuous steering instead of discrete bucket.

Pros:

- smoother steering.
- closer to vehicle LCC and trajectory-following.

Cons:

- requires reliable mouse/controller scale calibration.
- harder to evaluate offline.
- should come after the discrete LCC baseline.

## Recommended Roadmap

### Phase 1: LCC v1 Dataset

Collect `pathfinding_data_lcc_v1`.

Target:

- 30-50 short clips.
- 5-10 seconds each.
- multiple scenes, not only Tiger Vanguard route.

Suggested mix:

- 30% centered corridor keeping.
- 25% left/right wall correction.
- 20% wall-touch recovery.
- 15% narrow entrance alignment.
- 10% bends and forks.

Quality target:

- mouse active above 35%.
- forward below 75%.
- both left and right correction present.
- minimal idle or combat.

### Phase 2: LCC v1 Baseline

Prepare a manifest for LCC clips.

Train:

- dedicated v56-history LCC baseline.
- compare with current route policy plus v61 rule gate.

Metrics:

- forward/turn ratio.
- left/right switch count.
- bucket distribution.
- number of seconds before stuck.
- ability to pass neutral corridors without a boss goal.

### Phase 3: LCC-Assisted Goal Navigation

Runtime composition:

```text
goal policy -> proposed route intent
LCC/gate -> local correction intent
controller -> fused action
```

Initial fusion can be rule based:

- if LCC predicts correction with high confidence, override goal policy for a
  short window.
- otherwise use goal policy.

### Phase 4: Temporal/Continuous Upgrade

Only after LCC data proves useful:

- train GRU/LSTM on LCC clips.
- test continuous steering output.
- evaluate whether temporal modeling beats 4-frame stack.

## Risks

- If LCC clips are too easy, the model will learn forward again.
- If LCC clips are all recovery, the model may over-turn and become unstable.
- If only one area is recorded, it becomes another route patch, not foundation.
- If mouse capture misses movement, samples become misleading.
- Offline accuracy is weakly correlated with closed-loop navigation.

## Practical Acceptance Criteria

The LCC baseline should be considered useful when it can:

- move for 30 seconds in a familiar corridor without obvious wall collision.
- recover from mild left/right offset in at least two different scenes.
- pass a narrow entrance after 3-5 short correction examples.
- reduce left/right switch count compared with v6.0 gate.
- avoid `100% bucket=3` collapse in correction states.

## Current Recommendation

Do not switch to a large VLA yet. The near-term foundation step is:

```text
collect LCC v1 clips -> train dedicated LCC BC -> add LCC-assisted gate
```

This gives a reusable navigation layer for Tiger Vanguard, Hundred-Eyed Daoist
Master, and later no-target roaming.
