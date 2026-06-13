#!/usr/bin/env python3
"""Analyze inference telemetry CSV files."""
import argparse
import csv
from collections import Counter
from pathlib import Path


def pct(part, total):
    return part / max(total, 1) * 100.0


def analyze(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    executed = [row for row in rows if row.get("executed_step") == "1"]
    skipped = [row for row in rows if row.get("executed_step") != "1"]
    actions = Counter(row["action_name"] for row in executed)
    buckets = Counter(int(row["mouse_bucket"]) for row in executed)
    raw_dx = [int(row["raw_mouse_dx"]) for row in executed]
    action_conf = [float(row["action_confidence"]) for row in rows]
    mouse_conf = [float(row["mouse_confidence"]) for row in rows]

    switches = 0
    last_turn = None
    for row in executed:
        action = row["action_name"]
        turn = action if action in ("turn_left", "turn_right") else None
        if turn and last_turn and turn != last_turn:
            switches += 1
        if turn:
            last_turn = turn

    left_like = sum(1 for dx in raw_dx if dx < 0)
    right_like = sum(1 for dx in raw_dx if dx > 0)
    straight_like = sum(1 for dx in raw_dx if dx == 0)

    print(f"\n=== {path.name} ===")
    print(f"rows={len(rows)} executed={len(executed)} skipped={len(skipped)} execution_rate={pct(len(executed), len(rows)):.1f}%")
    print("actions:")
    for key, value in actions.most_common():
        print(f"  {key}: {value} ({pct(value, len(executed)):.1f}%)")
    print("mouse_buckets:")
    for key in sorted(buckets):
        print(f"  {key}: {buckets[key]} ({pct(buckets[key], len(executed)):.1f}%)")
    print(
        "raw_dx_direction: "
        f"left={left_like} ({pct(left_like, len(executed)):.1f}%) "
        f"straight={straight_like} ({pct(straight_like, len(executed)):.1f}%) "
        f"right={right_like} ({pct(right_like, len(executed)):.1f}%)"
    )
    print(f"turn_left/right_switches={switches}")
    if action_conf:
        print(f"avg_action_conf={sum(action_conf)/len(action_conf):.3f} avg_mouse_conf={sum(mouse_conf)/len(mouse_conf):.3f}")
    if rows and "gate_mode" in rows[0]:
        gate_modes = Counter(row.get("gate_mode", "") or "none" for row in rows)
        print("gate_modes:")
        for key, value in gate_modes.most_common():
            print(f"  {key}: {value} ({pct(value, len(rows)):.1f}%)")
    if rows and "policy_source" in rows[0]:
        sources = Counter(row.get("policy_source", "") or "unknown" for row in rows)
        details = Counter(row.get("policy_detail", "") or "none" for row in rows)
        print("policy_sources:")
        for key, value in sources.most_common():
            print(f"  {key}: {value} ({pct(value, len(rows)):.1f}%)")
        print("policy_details:")
        for key, value in details.most_common():
            print(f"  {key}: {value} ({pct(value, len(rows)):.1f}%)")

    flags = []
    forward_pct = pct(actions.get("forward", 0), len(executed))
    turn_pct = pct(actions.get("turn_left", 0) + actions.get("turn_right", 0), len(executed))
    if forward_pct < 25:
        flags.append("forward too low: likely not moving enough")
    if turn_pct > 70:
        flags.append("turning dominates: likely spinning/over-turning")
    if switches > max(8, len(executed) * 0.08):
        flags.append("frequent left/right switching: add smoothing or hysteresis")
    if pct(len(skipped), len(rows)) > 50:
        flags.append("many skipped low-confidence frames: lower threshold or improve model confidence")
    if flags:
        print("flags:")
        for flag in flags:
            print(f"  - {flag}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Wukong AI telemetry CSV")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for item in args.paths:
        analyze(Path(item))


if __name__ == "__main__":
    main()
