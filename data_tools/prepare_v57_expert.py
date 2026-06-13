#!/usr/bin/env python3
"""Prepare v5.7 expert keyboard/mouse recordings for v5.6 history training.

The collector records keyboard actions in the legacy action space, but v5.6
derives navigation labels from mouse_dx. This tool keeps the original mouse
signal, forces a chosen goal_id, filters unusable recordings, and writes a
merged manifest that can be passed directly to train_v56_history_stack_clean.py.
"""
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
class ExpertFileReport:
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


def count_values(values) -> dict[int, int]:
    unique, counts = np.unique(np.asarray(values).astype(int), return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def inspect_and_copy(
    source: Path,
    prepared_dir: Path,
    target_goal_id: int,
    mouse_dx_scale: float,
    min_frames: int,
    min_mouse_active_pct: float,
    min_forward_pct: float,
) -> ExpertFileReport:
    try:
        with h5py.File(source, "r") as hf:
            missing = [name for name in REQUIRED_DATASETS if name not in hf]
            if missing:
                return ExpertFileReport(str(source), None, False, f"missing:{','.join(missing)}")

            frames = int(len(hf["frames"]))
            usable = max(0, frames - LOOKBACK)
            if frames < min_frames:
                return ExpertFileReport(str(source), None, False, "too_short", frames, usable)
            if usable <= 0:
                return ExpertFileReport(str(source), None, False, "not_enough_for_history", frames, usable)

            mouse_dx = np.asarray(hf["mouse_dx"][:], dtype=np.float32)
            mouse_dy = np.asarray(hf["mouse_dy"][:], dtype=np.float32)
            active = (np.abs(mouse_dx) > 1) | (np.abs(mouse_dy) > 1)
            mouse_active_pct = float(active.sum() / max(frames, 1) * 100.0)
            if mouse_active_pct < min_mouse_active_pct:
                return ExpertFileReport(
                    str(source), None, False, "low_mouse_active", frames, usable, mouse_active_pct
                )

            actions = np.asarray(hf["actions"][:], dtype=np.int8)
            forward_pct = float(np.sum(actions == 4) / max(frames, 1) * 100.0)
            if forward_pct < min_forward_pct:
                return ExpertFileReport(
                    str(source), None, False, "low_forward", frames, usable, mouse_active_pct, forward_pct
                )

            scaled_mouse_dx = mouse_dx * mouse_dx_scale
            target_dx = scaled_mouse_dx[LOOKBACK:]
            action_labels = Counter(action_from_dx(float(x)) for x in target_dx)
            mouse_buckets = Counter(classify_bucket(float(x)) for x in target_dx)

            prepared_dir.mkdir(parents=True, exist_ok=True)
            prepared = prepared_dir / f"{source.stem}_goal{target_goal_id}{source.suffix}"
            if prepared.exists():
                prepared = prepared_dir / f"{source.stem}_goal{target_goal_id}_{int(time.time())}{source.suffix}"

            shutil.copy2(source, prepared)
            with h5py.File(prepared, "a") as out:
                if "goal_ids" in out:
                    del out["goal_ids"]
                if mouse_dx_scale != 1.0:
                    del out["mouse_dx"]
                    out.create_dataset("mouse_dx", data=scaled_mouse_dx.astype(np.float32))
                out.create_dataset("goal_ids", data=np.full(frames, target_goal_id, dtype=np.int8))
                out.attrs["prepared_by"] = "data_tools/prepare_v57_expert.py"
                out.attrs["prepared_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                out.attrs["forced_goal_id"] = int(target_goal_id)
                out.attrs["mouse_dx_scale"] = float(mouse_dx_scale)
                out.attrs["original_file"] = str(source)

            return ExpertFileReport(
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
        return ExpertFileReport(str(source), None, False, f"read_error:{exc}")


def load_base_files(base_manifest: Path | None) -> list[str]:
    if not base_manifest:
        return []
    payload = json.loads(base_manifest.read_text(encoding="utf-8"))
    return [str(item) for item in payload.get("accepted_files", [])]


def summarize(reports: list[ExpertFileReport], accepted_files: list[str]) -> dict:
    accepted = [r for r in reports if r.accepted]
    rejected = [r for r in reports if not r.accepted]
    buckets = Counter()
    actions = Counter()
    for report in accepted:
        buckets.update(report.mouse_buckets or {})
        actions.update(report.action_labels or {})
    frames = sum(r.frames for r in accepted)
    return {
        "expert_accepted_files": len(accepted),
        "expert_rejected_files": len(rejected),
        "expert_frames": frames,
        "expert_usable_samples": sum(r.usable_samples for r in accepted),
        "expert_mouse_active_pct": (
            sum(r.mouse_active_pct * r.frames for r in accepted) / max(frames, 1)
        ),
        "expert_forward_pct": sum(r.forward_pct * r.frames for r in accepted) / max(frames, 1),
        "expert_action_labels_from_dx": dict(sorted((int(k), int(v)) for k, v in actions.items())),
        "expert_mouse_buckets": dict(sorted((int(k), int(v)) for k, v in buckets.items())),
        "merged_accepted_files": len(accepted_files),
        "rejection_reasons": dict(Counter(r.reason for r in rejected)),
    }


def write_manifest(
    output_manifest: Path,
    base_manifest: Path | None,
    expert_dir: Path,
    prepared_dir: Path,
    target_goal_id: int,
    mouse_dx_scale: float,
    reports: list[ExpertFileReport],
    accepted_files: list[str],
) -> None:
    payload = {
        "source": "v57_expert_augmented",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_manifest": str(base_manifest) if base_manifest else None,
        "expert_source_dir": str(expert_dir),
        "expert_prepared_dir": str(prepared_dir),
        "forced_expert_goal_id": target_goal_id,
        "expert_mouse_dx_scale": mouse_dx_scale,
        "frame_stack": [-7, -3, -1, 0],
        "summary": summarize(reports, accepted_files),
        "accepted_files": accepted_files,
        "expert_files": [asdict(r) for r in reports],
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare v5.7 expert recordings")
    parser.add_argument("--expert-dir", default="pathfinding_data_v57_expert")
    parser.add_argument("--prepared-dir", default="pathfinding_data_v57_expert_goal1")
    parser.add_argument("--base-manifest", default="dataset_manifests/pathfinding_balanced_curated_v1_after_apply.json")
    parser.add_argument("--no-base-manifest", action="store_true", help="Write a manifest containing only expert files.")
    parser.add_argument("--output-manifest", default="dataset_manifests/pathfinding_v57_expert_augmented_goal1.json")
    parser.add_argument("--target-goal-id", type=int, default=1)
    parser.add_argument(
        "--mouse-dx-scale",
        type=float,
        default=1.0,
        help="Scale expert mouse_dx before training label derivation.",
    )
    parser.add_argument("--min-frames", type=int, default=500)
    parser.add_argument("--min-mouse-active-pct", type=float, default=10.0)
    parser.add_argument("--min-forward-pct", type=float, default=50.0)
    parser.add_argument("--replace-prepared", action="store_true")
    args = parser.parse_args()

    expert_dir = Path(args.expert_dir)
    prepared_dir = Path(args.prepared_dir)
    if args.replace_prepared and prepared_dir.exists():
        shutil.rmtree(prepared_dir)

    reports = [
        inspect_and_copy(
            source=path,
            prepared_dir=prepared_dir,
            target_goal_id=args.target_goal_id,
            mouse_dx_scale=args.mouse_dx_scale,
            min_frames=args.min_frames,
            min_mouse_active_pct=args.min_mouse_active_pct,
            min_forward_pct=args.min_forward_pct,
        )
        for path in sorted(expert_dir.glob("*.h5"))
    ]

    base_manifest = None if args.no_base_manifest else (Path(args.base_manifest) if args.base_manifest else None)
    accepted_files = load_base_files(base_manifest)
    accepted_files.extend(str(r.prepared) for r in reports if r.accepted and r.prepared)

    write_manifest(
        output_manifest=Path(args.output_manifest),
        base_manifest=base_manifest,
        expert_dir=expert_dir,
        prepared_dir=prepared_dir,
        target_goal_id=args.target_goal_id,
        mouse_dx_scale=args.mouse_dx_scale,
        reports=reports,
        accepted_files=accepted_files,
    )

    summary = summarize(reports, accepted_files)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"manifest={args.output_manifest}", flush=True)
    print(f"prepared_dir={args.prepared_dir}", flush=True)


if __name__ == "__main__":
    main()
