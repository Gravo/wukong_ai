#!/usr/bin/env python3
"""Train a cleaner v5.5 baseline from a curated manifest.

Differences from goal_conditioned_bc_v55_optimized.py:
  - Reads an explicit dataset manifest.
  - Splits by H5 file/episode, not random frame samples.
  - Keeps a held-out test split for a more honest baseline.
"""
import argparse
import importlib.util
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def load_v55_training_module():
    module_path = Path(__file__).resolve().parent / "goal_conditioned_bc_v55_optimized.py"
    spec = importlib.util.spec_from_file_location("goal_conditioned_bc_v55_optimized", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v55 = load_v55_training_module()
FRAME_STACK = _v55.FRAME_STACK
GRADIENT_ACCUMULATION_STEPS = _v55.GRADIENT_ACCUMULATION_STEPS
FocalLoss = _v55.FocalLoss
GoalConditionedBC_v55 = _v55.GoalConditionedBC_v55


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


def load_manifest(path: Path) -> list[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Path(item) for item in payload["accepted_files"]]


def file_goal_ids(path: Path) -> tuple[int, ...]:
    with h5py.File(path, "r") as hf:
        if "goal_ids" not in hf:
            return (0,)
        return tuple(sorted(set(np.asarray(hf["goal_ids"][:]).astype(int).tolist())))


def split_files(files: list[Path], val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for path in files:
        groups[file_goal_ids(path)].append(path)

    train, val, test = [], [], []
    for _, group_files in sorted(groups.items(), key=lambda item: item[0]):
        group_files = list(group_files)
        rng.shuffle(group_files)
        n = len(group_files)
        n_test = max(1, round(n * test_ratio)) if n >= 3 and test_ratio > 0 else 0
        n_val = max(1, round(n * val_ratio)) if n - n_test >= 3 and val_ratio > 0 else 0
        test.extend(group_files[:n_test])
        val.extend(group_files[n_test : n_test + n_val])
        train.extend(group_files[n_test + n_val :])

    return train, val, test


class ManifestV55Dataset(Dataset):
    """Preloaded v5.5 dataset for a selected list of H5 episode files."""

    def __init__(self, files: list[Path], verbose=True):
        self.files = files
        self.samples = []
        self.frames_cache = {}
        self.mouse_dx_cache = {}
        self.goal_ids_cache = {}

        for file_idx, h5_file in enumerate(tqdm(files, desc="preload", disable=not verbose)):
            with h5py.File(h5_file, "r") as hf:
                n = len(hf["frames"])
                self.frames_cache[file_idx] = hf["frames"][:]
                self.mouse_dx_cache[file_idx] = hf["mouse_dx"][:]
                if "goal_ids" in hf:
                    self.goal_ids_cache[file_idx] = hf["goal_ids"][:]
                else:
                    self.goal_ids_cache[file_idx] = np.zeros(n, dtype=np.int64)
                for frame_idx in range(n - max(FRAME_STACK) - 1):
                    self.samples.append((file_idx, frame_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, frame_idx = self.samples[idx]
        frames = []
        for offset in FRAME_STACK:
            frame = self.frames_cache[file_idx][frame_idx + offset]
            frame = torch.from_numpy(frame).float() / 255.0
            frames.append(frame.permute(2, 0, 1))

        mouse_dx = self.mouse_dx_cache[file_idx][frame_idx + FRAME_STACK[-1]]
        goal_id = int(self.goal_ids_cache[file_idx][frame_idx + FRAME_STACK[-1]])
        return (
            torch.cat(frames, dim=0),
            torch.tensor(action_from_dx(mouse_dx), dtype=torch.long),
            torch.tensor(classify_bucket(mouse_dx), dtype=torch.long),
            torch.tensor(goal_id, dtype=torch.long),
        )

    def label_summary(self):
        actions = Counter()
        buckets = Counter()
        goals = Counter()
        for idx in range(len(self.samples)):
            _, action, bucket, goal = self[idx]
            actions[int(action)] += 1
            buckets[int(bucket)] += 1
            goals[int(goal)] += 1
        return {"actions": dict(actions), "mouse_buckets": dict(buckets), "goals": dict(goals)}


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def evaluate(model, loader, action_loss_fn, mouse_loss_fn, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct_action = 0
    correct_mouse = 0
    with torch.no_grad():
        for frames, actions, buckets, goals in loader:
            frames = frames.to(device)
            actions = actions.to(device)
            buckets = buckets.to(device)
            goals = goals.to(device)
            action_logits, mouse_logits = model(frames, goals)
            loss = action_loss_fn(action_logits, actions) + mouse_loss_fn(mouse_logits, buckets)
            total_loss += float(loss.item())
            correct_action += int(action_logits.argmax(1).eq(actions).sum().item())
            correct_mouse += int(mouse_logits.argmax(1).eq(buckets).sum().item())
            total += int(actions.size(0))
    return {
        "loss": total_loss / max(len(loader), 1),
        "acc_action": correct_action / max(total, 1) * 100.0,
        "acc_mouse": correct_mouse / max(total, 1) * 100.0,
        "samples": total,
    }


def write_split_manifest(path: Path, train, val, test, stats):
    payload = {
        "train_files": [str(p) for p in train],
        "val_files": [str(p) for p in val],
        "test_files": [str(p) for p in test],
        "stats": stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_status(path: Path, **fields):
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **fields,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train(args):
    manifest_path = Path(args.manifest)
    files = load_manifest(manifest_path)
    train_files, val_files, test_files = split_files(files, args.val_ratio, args.test_ratio, args.seed)

    print(f"[split] train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    for name, split in [("train", train_files), ("val", val_files), ("test", test_files)]:
        print(f"[split] {name}:")
        for item in split:
            print(f"  - {item.name} goals={file_goal_ids(item)}")

    train_ds = ManifestV55Dataset(train_files, verbose=not args.quiet)
    val_ds = ManifestV55Dataset(val_files, verbose=not args.quiet)
    test_ds = ManifestV55Dataset(test_files, verbose=not args.quiet)

    stats = {
        "train": train_ds.label_summary(),
        "val": val_ds.label_summary(),
        "test": test_ds.label_summary(),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifest(output_dir / "v55_clean_split_manifest.json", train_files, val_files, test_files, stats)
    status_path = output_dir / "status_v55_clean.json"

    if args.dry_run:
        print("[dry-run] split and label summary complete; no training started.")
        write_status(status_path, status="dry_run_complete", epochs_requested=0)
        return

    run_start = time.time()
    write_status(
        status_path,
        status="running",
        current_epoch=0,
        epochs_requested=args.epochs,
        manifest=str(manifest_path),
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GoalConditionedBC_v55(num_goals=args.num_goals, freeze_backbone=args.freeze_backbone).to(device)
    action_weights = torch.tensor([1.0, 8.0, 8.0], device=device)
    mouse_weights = torch.tensor([1.5, 2.0, 2.0, 1.0, 2.0, 2.0, 1.5], device=device)
    action_loss_fn = FocalLoss(weight=action_weights, gamma=2.0)
    mouse_loss_fn = FocalLoss(weight=mouse_weights, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False)

    best_val = float("inf")
    log_path = output_dir / "training_log_v55_clean.csv"
    log_path.write_text("epoch,train_loss,val_loss,val_acc_action,val_acc_mouse,lr,time\n", encoding="utf-8")

    try:
        for epoch in range(args.epochs):
            model.train()
            start = time.time()
            total_loss = 0.0
            optimizer.zero_grad()
            if args.progress == "epoch":
                print(f"[epoch {epoch + 1}/{args.epochs}] started", flush=True)
            train_iter = (
                tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
                if args.progress == "batch"
                else train_loader
            )
            for batch_idx, (frames, actions, buckets, goals) in enumerate(train_iter):
                frames = frames.to(device)
                actions = actions.to(device)
                buckets = buckets.to(device)
                goals = goals.to(device)
                action_logits, mouse_logits = model(frames, goals)
                loss = (action_loss_fn(action_logits, actions) + mouse_loss_fn(mouse_logits, buckets))
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                batch_loss = float(loss.item()) * GRADIENT_ACCUMULATION_STEPS
                total_loss += batch_loss
                if args.progress == "batch":
                    train_iter.set_postfix({"loss": f"{batch_loss:.4f}"})

            if len(train_loader) % GRADIENT_ACCUMULATION_STEPS:
                optimizer.step()
                optimizer.zero_grad()

            val_metrics = evaluate(model, val_loader, action_loss_fn, mouse_loss_fn, device)
            scheduler.step()
            train_loss = total_loss / max(len(train_loader), 1)
            elapsed = time.time() - start
            lr = scheduler.get_last_lr()[0]
            print(
                f"[epoch {epoch + 1}/{args.epochs}] train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc_a={val_metrics['acc_action']:.2f}% "
                f"val_acc_m={val_metrics['acc_mouse']:.2f}% "
                f"time={elapsed:.1f}s",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"{epoch + 1},{train_loss:.4f},{val_metrics['loss']:.4f},"
                    f"{val_metrics['acc_action']:.2f},{val_metrics['acc_mouse']:.2f},{lr:.8f},{elapsed:.2f}\n"
                )
            write_status(
                status_path,
                status="running",
                current_epoch=epoch + 1,
                epochs_requested=args.epochs,
                train_loss=train_loss,
                val_metrics=val_metrics,
                elapsed_seconds=time.time() - run_start,
            )
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "manifest": str(manifest_path),
                    },
                    output_dir / "goal_bc_v55_clean_best.pt",
                )

        checkpoint = torch.load(output_dir / "goal_bc_v55_clean_best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, action_loss_fn, mouse_loss_fn, device)
        (output_dir / "test_metrics_v55_clean.json").write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_status(
            status_path,
            status="complete",
            current_epoch=args.epochs,
            epochs_requested=args.epochs,
            best_val_loss=best_val,
            test_metrics=test_metrics,
            elapsed_seconds=time.time() - run_start,
        )
        print("[test]", json.dumps(test_metrics, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        write_status(
            status_path,
            status="failed",
            current_epoch=None,
            epochs_requested=args.epochs,
            error=str(exc),
            elapsed_seconds=time.time() - run_start,
        )
        raise


def main():
    parser = argparse.ArgumentParser(description="Train clean v5.5 baseline")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="checkpoints/v55_clean_baseline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-goals", type=int, default=2)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--progress",
        choices=["epoch", "batch", "none"],
        default="epoch",
        help="Default avoids noisy batch-level tqdm output.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
