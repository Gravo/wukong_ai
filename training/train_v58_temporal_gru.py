#!/usr/bin/env python3
"""Train a v5.8 GRU temporal baseline on the same manifest semantics as v5.6."""
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.temporal_gru_bc import GoalConditionedGRU
from training.goal_conditioned_bc_v55_optimized import FocalLoss, GRADIENT_ACCUMULATION_STEPS
from training.train_v56_history_stack_clean import (
    action_from_dx,
    classify_bucket,
    evaluate,
    file_goal_ids,
    load_manifest,
    split_files,
    write_split_manifest,
    write_status,
)


class ManifestV58TemporalDataset(Dataset):
    """Preloaded sequence dataset: frames[t-seq+1:t] -> label[t]."""

    def __init__(self, files: list[Path], seq_length: int, verbose=True):
        self.files = files
        self.seq_length = seq_length
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
                for frame_idx in range(seq_length - 1, n):
                    self.samples.append((file_idx, frame_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, frame_idx = self.samples[idx]
        start = frame_idx - self.seq_length + 1
        frames_np = self.frames_cache[file_idx][start : frame_idx + 1]
        frames = torch.from_numpy(frames_np).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)

        mouse_dx = float(self.mouse_dx_cache[file_idx][frame_idx])
        goal_id = int(self.goal_ids_cache[file_idx][frame_idx])
        return (
            frames,
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


def save_checkpoint(path, model, optimizer, epoch, val_metrics, manifest_path, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "manifest": str(manifest_path),
            "model_config": {
                "num_goals": args.num_goals,
                "seq_length": args.seq_length,
                "hidden_dim": args.hidden_dim,
                "gru_layers": args.gru_layers,
                "dropout": args.dropout,
            },
        },
        path,
    )


def train(args):
    manifest_path = Path(args.manifest)
    files = load_manifest(manifest_path)
    train_files, val_files, test_files = split_files(files, args.val_ratio, args.test_ratio, args.seed)

    print(f"[split] train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    for name, split in [("train", train_files), ("val", val_files), ("test", test_files)]:
        print(f"[split] {name}:")
        for item in split:
            print(f"  - {item.name} goals={file_goal_ids(item)}")

    train_ds = ManifestV58TemporalDataset(train_files, args.seq_length, verbose=not args.quiet)
    val_ds = ManifestV58TemporalDataset(val_files, args.seq_length, verbose=not args.quiet)
    test_ds = ManifestV58TemporalDataset(test_files, args.seq_length, verbose=not args.quiet)
    stats = {
        "train": train_ds.label_summary(),
        "val": val_ds.label_summary(),
        "test": test_ds.label_summary(),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifest(output_dir / "v58_temporal_split_manifest.json", train_files, val_files, test_files, stats)
    status_path = output_dir / "status_v58_temporal.json"
    if args.dry_run:
        write_status(status_path, status="dry_run_complete", epochs_requested=0)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GoalConditionedGRU(
        num_goals=args.num_goals,
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    action_weights = torch.tensor([1.0, 2.0, 2.0], device=device)
    mouse_weights = torch.tensor([1.5, 2.0, 2.0, 1.0, 2.0, 2.0, 1.5], device=device)
    action_loss_fn = FocalLoss(weight=action_weights, gamma=2.0)
    mouse_loss_fn = FocalLoss(weight=mouse_weights, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False)

    best_val = float("inf")
    best_action = -1.0
    run_start = time.time()
    log_path = output_dir / "training_log_v58_temporal.csv"
    log_path.write_text("epoch,train_loss,val_loss,val_acc_action,val_acc_mouse,lr,time\n", encoding="utf-8")
    write_status(status_path, status="running", current_epoch=0, epochs_requested=args.epochs)

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
            save_checkpoint(output_dir / "goal_bc_v58_temporal_last.pt", model, optimizer, epoch + 1, val_metrics, manifest_path, args)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(output_dir / "goal_bc_v58_temporal_best.pt", model, optimizer, epoch + 1, val_metrics, manifest_path, args)
            if val_metrics["acc_action"] > best_action:
                best_action = val_metrics["acc_action"]
                save_checkpoint(output_dir / "goal_bc_v58_temporal_best_action.pt", model, optimizer, epoch + 1, val_metrics, manifest_path, args)

        checkpoint = torch.load(output_dir / "goal_bc_v58_temporal_best_action.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, action_loss_fn, mouse_loss_fn, device)
        (output_dir / "test_metrics_v58_temporal.json").write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_status(
            status_path,
            status="complete",
            current_epoch=args.epochs,
            epochs_requested=args.epochs,
            best_val_loss=best_val,
            best_val_action=best_action,
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
    parser = argparse.ArgumentParser(description="Train v5.8 temporal GRU baseline")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="checkpoints/v58_temporal_gru_baseline")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-goals", type=int, default=2)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progress", choices=["epoch", "batch", "none"], default="epoch")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
