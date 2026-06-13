#!/usr/bin/env python3
"""Inspect and curate H5 pathfinding datasets.

The tool writes a manifest for accepted files and can optionally move rejected
files into a quarantine directory. It is intentionally conservative: rejected
files are not deleted.
"""
import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np


REQUIRED_DATASETS = ("frames", "actions", "mouse_dx", "mouse_dy")
FRAME_STACK = (0, 1, 3, 7)


@dataclass
class FileQuality:
    path: str
    file_name: str
    accepted: bool
    reason: str
    frames: int = 0
    usable_samples: int = 0
    mouse_active_frames: int = 0
    mouse_active_pct: float = 0.0
    goal_ids: list[int] | None = None
    recorded_actions: dict[int, int] | None = None
    mouse_buckets: dict[int, int] | None = None


def classify_bucket(mouse_dx: float) -> int:
    if mouse_dx <= -200:
        return 0
    if mouse_dx <= -100:
        return 1
    if mouse_dx <= -20:
        return 2
    if mouse_dx <= 20:
        return 3
    if mouse_dx <= 100:
        return 4
    if mouse_dx <= 200:
        return 5
    return 6


def count_values(values) -> dict[int, int]:
    unique, counts = np.unique(np.asarray(values).astype(int), return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def count_labels(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def inspect_file(path: Path, min_frames: int) -> FileQuality:
    try:
        with h5py.File(path, "r") as hf:
            missing = [name for name in REQUIRED_DATASETS if name not in hf]
            if missing:
                return FileQuality(str(path), path.name, False, f"missing:{','.join(missing)}")

            frames = int(len(hf["frames"]))
            usable = max(0, frames - max(FRAME_STACK) - 1)
            if frames < min_frames:
                return FileQuality(str(path), path.name, False, "too_short", frames, usable)
            if frames <= max(FRAME_STACK) + 1:
                return FileQuality(str(path), path.name, False, "not_enough_for_stack", frames, usable)

            mouse_dx = np.asarray(hf["mouse_dx"][:])
            mouse_dy = np.asarray(hf["mouse_dy"][:])
            active = (np.abs(mouse_dx) > 1) | (np.abs(mouse_dy) > 1)
            active_count = int(active.sum())
            active_pct = active_count / max(frames, 1) * 100.0
            if active_count == 0:
                return FileQuality(
                    str(path),
                    path.name,
                    False,
                    "zero_mouse_active",
                    frames,
                    usable,
                    active_count,
                    active_pct,
                )

            actions = np.asarray(hf["actions"][:])
            goal_ids = np.asarray(hf["goal_ids"][:]).astype(int) if "goal_ids" in hf else np.zeros(frames, dtype=int)
            target_dx = mouse_dx[max(FRAME_STACK) : frames - 1]
            buckets = [classify_bucket(x) for x in target_dx]

            return FileQuality(
                path=str(path),
                file_name=path.name,
                accepted=True,
                reason="accepted",
                frames=frames,
                usable_samples=usable,
                mouse_active_frames=active_count,
                mouse_active_pct=active_pct,
                goal_ids=sorted(int(x) for x in set(goal_ids.tolist())),
                recorded_actions=count_values(actions),
                mouse_buckets=count_values(buckets),
            )
    except Exception as exc:
        return FileQuality(str(path), path.name, False, f"read_error:{exc}")


def summarize(results: list[FileQuality]) -> dict:
    accepted = [r for r in results if r.accepted]
    rejected = [r for r in results if not r.accepted]
    bucket_counts: dict[int, int] = {}
    goal_file_counts: dict[int, int] = {}
    frames = sum(r.frames for r in accepted)
    samples = sum(r.usable_samples for r in accepted)
    mouse_active = sum(r.mouse_active_frames for r in accepted)

    for result in accepted:
        for bucket, count in (result.mouse_buckets or {}).items():
            bucket_counts[int(bucket)] = bucket_counts.get(int(bucket), 0) + int(count)
        for goal_id in result.goal_ids or []:
            goal_file_counts[int(goal_id)] = goal_file_counts.get(int(goal_id), 0) + 1

    return {
        "accepted_files": len(accepted),
        "rejected_files": len(rejected),
        "accepted_frames": frames,
        "usable_samples": samples,
        "mouse_active_frames": mouse_active,
        "mouse_active_pct": mouse_active / max(frames, 1) * 100.0,
        "mouse_bucket_counts": dict(sorted(bucket_counts.items())),
        "goal_file_presence": dict(sorted(goal_file_counts.items())),
        "rejection_reasons": count_labels([r.reason for r in rejected]) if rejected else {},
    }


def move_rejected(results: list[FileQuality], source_dir: Path, quarantine_dir: Path) -> None:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        if result.accepted:
            continue
        src = Path(result.path)
        if not src.exists():
            continue
        reason_dir = quarantine_dir / result.reason.replace(":", "_")
        reason_dir.mkdir(parents=True, exist_ok=True)
        dst = reason_dir / src.name
        if dst.exists():
            dst = reason_dir / f"{src.stem}_{int(time.time())}{src.suffix}"
        shutil.move(str(src), str(dst))


def write_manifest(results: list[FileQuality], source_dir: Path, output: Path) -> None:
    accepted = [r for r in results if r.accepted]
    payload = {
        "source_dir": str(source_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_stack": list(FRAME_STACK),
        "summary": summarize(results),
        "accepted_files": [r.path for r in accepted],
        "files": [asdict(r) for r in results],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Curate H5 pathfinding datasets")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--quarantine-dir", default="data_quarantine")
    parser.add_argument("--min-frames", type=int, default=100)
    parser.add_argument("--apply", action="store_true", help="Move rejected files to quarantine")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    results = [inspect_file(path, args.min_frames) for path in sorted(source_dir.glob("*.h5"))]
    write_manifest(results, source_dir, Path(args.manifest))

    if args.apply:
        move_rejected(results, source_dir, Path(args.quarantine_dir))

    summary = summarize(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"manifest={args.manifest}")
    if args.apply:
        print(f"quarantine={args.quarantine_dir}")


if __name__ == "__main__":
    main()
