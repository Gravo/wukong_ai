#!/usr/bin/env python3
"""Prepare command-conditioned LCC clips.

Command labels can be inferred from the parent directory name:

  pathfinding_data_lcc_cmd_v1/KEEP_CENTER/*.h5
  pathfinding_data_lcc_cmd_v1/AVOID_RIGHT_WALL/*.h5

or overridden by a JSON map:

  {"some_file.h5": "TURN_LEFT_SOON", "other_stem": "RECOVER_FROM_STUCK"}
"""
import argparse
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

from core.lcc_commands import LCC_COMMANDS, command_id, normalize_command_name
from data_tools.prepare_lcc_dataset import action_from_dx, classify_bucket


REQUIRED_DATASETS = ("frames", "actions", "mouse_dx", "mouse_dy")
LOOKBACK = 7


@dataclass
class CommandLccFileReport:
    source: str
    prepared: str | None
    accepted: bool
    reason: str
    command_id: int | None = None
    command_name: str | None = None
    frames: int = 0
    usable_samples: int = 0
    mouse_active_pct: float = 0.0
    forward_pct: float = 0.0
    action_labels: dict[int, int] | None = None
    mouse_buckets: dict[int, int] | None = None


def load_command_map(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in payload.items()}


def infer_command(path: Path, source_dir: Path, command_map: dict[str, str], default_command: str | None):
    for key in (path.name, path.stem, str(path)):
        if key in command_map:
            return command_id(command_map[key])

    try:
        relative = path.relative_to(source_dir)
        if len(relative.parts) > 1:
            parent = normalize_command_name(relative.parts[0])
            return command_id(parent)
    except ValueError:
        pass

    if default_command is not None:
        return command_id(default_command)
    raise ValueError("missing_command")


def inspect_and_copy(
    source: Path,
    source_dir: Path,
    prepared_dir: Path,
    command_map: dict[str, str],
    default_command: str | None,
    mouse_dx_scale: float,
    min_frames: int,
    min_mouse_active_pct: float,
    max_idle_pct: float,
) -> CommandLccFileReport:
    try:
        cmd_id = infer_command(source, source_dir, command_map, default_command)
        cmd_name = LCC_COMMANDS[cmd_id]
    except Exception:
        return CommandLccFileReport(str(source), None, False, "missing_or_invalid_command")

    try:
        with h5py.File(source, "r") as hf:
            missing = [name for name in REQUIRED_DATASETS if name not in hf]
            if missing:
                return CommandLccFileReport(str(source), None, False, f"missing:{','.join(missing)}", cmd_id, cmd_name)

            frames = int(len(hf["frames"]))
            usable = max(0, frames - LOOKBACK)
            if frames < min_frames:
                return CommandLccFileReport(str(source), None, False, "too_short", cmd_id, cmd_name, frames, usable)
            if usable <= 0:
                return CommandLccFileReport(str(source), None, False, "not_enough_for_history", cmd_id, cmd_name, frames, usable)

            mouse_dx = np.asarray(hf["mouse_dx"][:], dtype=np.float32)
            mouse_dy = np.asarray(hf["mouse_dy"][:], dtype=np.float32)
            active = (np.abs(mouse_dx) > 1) | (np.abs(mouse_dy) > 1)
            mouse_active_pct = float(active.sum() / max(frames, 1) * 100.0)
            if mouse_active_pct < min_mouse_active_pct:
                return CommandLccFileReport(
                    str(source), None, False, "low_mouse_active", cmd_id, cmd_name, frames, usable, mouse_active_pct
                )

            actions = np.asarray(hf["actions"][:], dtype=np.int8)
            idle_pct = float(np.sum(actions == 0) / max(frames, 1) * 100.0)
            if idle_pct > max_idle_pct:
                return CommandLccFileReport(
                    str(source), None, False, "high_idle", cmd_id, cmd_name, frames, usable, mouse_active_pct
                )

            forward_pct = float(np.sum(actions == 4) / max(frames, 1) * 100.0)
            scaled_mouse_dx = mouse_dx * mouse_dx_scale
            target_dx = scaled_mouse_dx[LOOKBACK:]
            action_labels = Counter(action_from_dx(float(x)) for x in target_dx)
            mouse_buckets = Counter(classify_bucket(float(x)) for x in target_dx)

            command_dir = prepared_dir / cmd_name
            command_dir.mkdir(parents=True, exist_ok=True)
            prepared = command_dir / f"{source.stem}_cmd{cmd_id}{source.suffix}"
            if prepared.exists():
                prepared = command_dir / f"{source.stem}_cmd{cmd_id}_{int(time.time())}{source.suffix}"

            shutil.copy2(source, prepared)
            with h5py.File(prepared, "a") as out:
                if "goal_ids" in out:
                    del out["goal_ids"]
                if "command_ids" in out:
                    del out["command_ids"]
                if mouse_dx_scale != 1.0:
                    del out["mouse_dx"]
                    out.create_dataset("mouse_dx", data=scaled_mouse_dx.astype(np.float32))
                command_values = np.full(frames, cmd_id, dtype=np.int8)
                out.create_dataset("goal_ids", data=command_values)
                out.create_dataset("command_ids", data=command_values)
                out.attrs["prepared_by"] = "data_tools/prepare_command_lcc_dataset.py"
                out.attrs["prepared_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                out.attrs["task"] = "command_lcc"
                out.attrs["command_id"] = int(cmd_id)
                out.attrs["command_name"] = cmd_name
                out.attrs["mouse_dx_scale"] = float(mouse_dx_scale)
                out.attrs["original_file"] = str(source)

            return CommandLccFileReport(
                source=str(source),
                prepared=str(prepared),
                accepted=True,
                reason="accepted",
                command_id=cmd_id,
                command_name=cmd_name,
                frames=frames,
                usable_samples=usable,
                mouse_active_pct=mouse_active_pct,
                forward_pct=forward_pct,
                action_labels=dict(sorted(action_labels.items())),
                mouse_buckets=dict(sorted(mouse_buckets.items())),
            )
    except Exception as exc:
        return CommandLccFileReport(str(source), None, False, f"read_error:{exc}", cmd_id, cmd_name)


def summarize(reports: list[CommandLccFileReport]) -> dict:
    accepted = [r for r in reports if r.accepted]
    rejected = [r for r in reports if not r.accepted]
    frames = sum(r.frames for r in accepted)
    actions = Counter()
    buckets = Counter()
    commands = Counter()
    for report in accepted:
        actions.update(report.action_labels or {})
        buckets.update(report.mouse_buckets or {})
        commands[report.command_name] += report.usable_samples
    return {
        "accepted_files": len(accepted),
        "rejected_files": len(rejected),
        "accepted_frames": frames,
        "usable_samples": sum(r.usable_samples for r in accepted),
        "mouse_active_pct": sum(r.mouse_active_pct * r.frames for r in accepted) / max(frames, 1),
        "forward_pct": sum(r.forward_pct * r.frames for r in accepted) / max(frames, 1),
        "command_sample_counts": dict(commands),
        "action_labels_from_dx": dict(sorted((int(k), int(v)) for k, v in actions.items())),
        "mouse_buckets": dict(sorted((int(k), int(v)) for k, v in buckets.items())),
        "rejection_reasons": dict(Counter(r.reason for r in rejected)),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare command-conditioned LCC clips")
    parser.add_argument("--source-dir", default="pathfinding_data_lcc_cmd_v1")
    parser.add_argument("--prepared-dir", default="pathfinding_data_lcc_cmd_v1_prepared")
    parser.add_argument("--manifest", default="dataset_manifests/pathfinding_lcc_cmd_v1.json")
    parser.add_argument("--command-map", default=None)
    parser.add_argument("--default-command", default=None)
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
    command_map = load_command_map(Path(args.command_map) if args.command_map else None)
    reports = [
        inspect_and_copy(
            path,
            source_dir,
            prepared_dir,
            command_map,
            args.default_command,
            args.mouse_dx_scale,
            args.min_frames,
            args.min_mouse_active_pct,
            args.max_idle_pct,
        )
        for path in sorted(source_dir.rglob("*.h5"))
    ]
    accepted_files = [str(r.prepared) for r in reports if r.accepted and r.prepared]
    payload = {
        "source": "command_lcc_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": str(source_dir),
        "prepared_dir": str(prepared_dir),
        "mouse_dx_scale": args.mouse_dx_scale,
        "commands": LCC_COMMANDS,
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
