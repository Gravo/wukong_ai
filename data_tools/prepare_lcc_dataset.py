#!/usr/bin/env python3
"""Prepare local corridor-keeping clips for the v6.2 LCC baseline."""
import argparse
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np


REQUIRED_DATASETS = ("frames", "actions", "mouse_dx", "mouse_dy")
LOOKBACK = 7


@dataclass
class LccFileReport:
    source: str
    prepared: str | None
    accepted: bool
    reason: str
    frames: int = 0
    usable_samples: int = 0
    mouse_active_pct: float = 0.0
    forward_pct: float = 0.0
    action_labels: dict[int, int] | None = None
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


def action_from_dx(mouse_dx: float) -> int:
    if mouse_dx < -20:
        return 1
    if mouse_dx > 20:
        return 2
    return 0


def inspect_and_copy(
    source: Path,
    prepared_dir: Path,
    mouse_dx_scale: float,
    min_frames: int,
    min_mouse_active_pct: float,
    max_idle_pct: float,
) -> LccFileReport:
    try:
        with h5py.File(source, "r") as hf:
            missing = [name for name in REQUIRED_DATASETS if name not in hf]
            if missing:
                return LccFileReport(str(source), None, False, f"missing:{','.join(missing)}")

            frames = int(len(hf["frames"]))
            usable = max(0, frames - LOOKBACK)
            if frames < min_frames:
                return LccFileReport(str(source), None, False, "too_short", frames, usable)
            if usable <= 0:
                return LccFileReport(str(source), None, False, "not_enough_for_history", frames, usable)

            mouse_dx = np.asarray(hf["mouse_dx"][:], dtype=np.float32)
            mouse_dy = np.asarray(hf["mouse_dy"][:], dtype=np.float32)
            active = (np.abs(mouse_dx) > 1) | (np.abs(mouse_dy) > 1)
            mouse_active_pct = float(active.sum() / max(frames, 1) * 100.0)
            if mouse_active_pct < min_mouse_active_pct:
                return LccFileReport(
                    str(source), None, False, "low_mouse_active", frames, usable, mouse_active_pct
                )

            actions = np.asarray(hf["actions"][:], dtype=np.int8)
            idle_pct = float(np.sum(actions == 0) / max(frames, 1) * 100.0)
            if idle_pct > max_idle_pct:
                return LccFileReport(
                    str(source), None, False, "high_idle", frames, usable, mouse_active_pct
                )

            forward_pct = float(np.sum(actions == 4) / max(frames, 1) * 100.0)
            scaled_mouse_dx = mouse_dx * mouse_dx_scale
            target_dx = scaled_mouse_dx[LOOKBACK:]
            action_labels = Counter(action_from_dx(float(x)) for x in target_dx)
            mouse_buckets = Counter(classify_bucket(float(x)) for x in target_dx)

            prepared_dir.mkdir(parents=True, exist_ok=True)
            prepared = prepared_dir / f"{source.stem}_lcc{source.suffix}"
            if prepared.exists():
                prepared = prepared_dir / f"{source.stem}_lcc_{int(time.time())}{source.suffix}"

            shutil.copy2(source, prepared)
            with h5py.File(prepared, "a") as out:
                if "goal_ids" in out:
                    del out["goal_ids"]
                if mouse_dx_scale != 1.0:
                    del out["mouse_dx"]
                    out.create_dataset("mouse_dx", data=scaled_mouse_dx.astype(np.float32))
                out.create_dataset("goal_ids", data=np.zeros(frames, dtype=np.int8))
                out.attrs["prepared_by"] = "data_tools/prepare_lcc_dataset.py"
                out.attrs["prepared_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                out.attrs["task"] = "lcc"
                out.attrs["mouse_dx_scale"] = float(mouse_dx_scale)
                out.attrs["original_file"] = str(source)

            return LccFileReport(
                source=str(source),
                prepared=str(prepared),
                accepted=True,
                reason="accepted",
                frames=frames,
                usable_samples=usable,
                mouse_active_pct=mouse_active_pct,
                forward_pct=forward_pct,
                action_labels=dict(sorted(action_labels.items())),
                mouse_buckets=dict(sorted(mouse_buckets.items())),
            )
    except Exception as exc:
        return LccFileReport(str(source), None, False, f"read_error:{exc}")


def summarize(reports: list[LccFileReport]) -> dict:
    accepted = [r for r in reports if r.accepted]
    rejected = [r for r in reports if not r.accepted]
    frames = sum(r.frames for r in accepted)
    actions = Counter()
    buckets = Counter()
    for report in accepted:
        actions.update(report.action_labels or {})
        buckets.update(report.mouse_buckets or {})
    return {
        "accepted_files": len(accepted),
        "rejected_files": len(rejected),
        "accepted_frames": frames,
        "usable_samples": sum(r.usable_samples for r in accepted),
        "mouse_active_pct": sum(r.mouse_active_pct * r.frames for r in accepted) / max(frames, 1),
        "forward_pct": sum(r.forward_pct * r.frames for r in accepted) / max(frames, 1),
        "action_labels_from_dx": dict(sorted((int(k), int(v)) for k, v in actions.items())),
        "mouse_buckets": dict(sorted((int(k), int(v)) for k, v in buckets.items())),
        "rejection_reasons": dict(Counter(r.reason for r in rejected)),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare LCC v1 clips")
    parser.add_argument("--source-dir", default="pathfinding_data_lcc_v1")
    parser.add_argument("--prepared-dir", default="pathfinding_data_lcc_v1_prepared")
    parser.add_argument("--manifest", default="dataset_manifests/pathfinding_lcc_v1.json")
    parser.add_argument("--mouse-dx-scale", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=45)
    parser.add_argument("--min-mouse-active-pct", type=float, default=10.0)
    parser.add_argument("--max-idle-pct", type=float, default=35.0)
    parser.add_argument("--replace-prepared", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    prepared_dir = Path(args.prepared_dir)
    if args.replace_prepared and prepared_dir.exists():
        shutil.rmtree(prepared_dir)

    reports = [
        inspect_and_copy(
            path,
            prepared_dir,
            args.mouse_dx_scale,
            args.min_frames,
            args.min_mouse_active_pct,
            args.max_idle_pct,
        )
        for path in sorted(source_dir.glob("*.h5"))
    ]
    accepted_files = [str(r.prepared) for r in reports if r.accepted and r.prepared]
    payload = {
        "source": "lcc_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": str(source_dir),
        "prepared_dir": str(prepared_dir),
        "mouse_dx_scale": args.mouse_dx_scale,
        "frame_stack": [-7, -3, -1, 0],
        "summary": summarize(reports),
        "accepted_files": accepted_files,
        "files": [asdict(r) for r in reports],
    }
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"manifest={manifest}", flush=True)
    print(f"prepared_dir={prepared_dir}", flush=True)


if __name__ == "__main__":
    main()
