#!/usr/bin/env python3
"""Train v5.9 DAgger-style history policy with forced weighted correction data."""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import ConcatDataset, DataLoader

from training.train_v56_history_stack_clean import (
    FocalLoss,
    GRADIENT_ACCUMULATION_STEPS,
    GoalConditionedBC_v55,
    ManifestV56HistoryDataset,
    evaluate,
    file_goal_ids,
    load_manifest,
    split_files,
    write_split_manifest,
    write_status,
)


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def concat_label_summary(datasets):
    actions = Counter()
    buckets = Counter()
    goals = Counter()
    for dataset in datasets:
        summary = dataset.label_summary()
        actions.update(summary["actions"])
        buckets.update(summary["mouse_buckets"])
        goals.update(summary["goals"])
    return {"actions": dict(actions), "mouse_buckets": dict(buckets), "goals": dict(goals)}


def save_checkpoint(path, model, optimizer, epoch, val_metrics, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "manifest": str(args.manifest),
            "extra_train_manifest": str(args.extra_train_manifest),
            "extra_repeat": args.extra_repeat,
            "training_mode": "v59_dagger_history",
        },
        path,
    )


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path = Path(args.manifest)
    base_files = load_manifest(manifest_path)
    train_files, val_files, test_files = split_files(base_files, args.val_ratio, args.test_ratio, args.seed)
    extra_files = load_manifest(Path(args.extra_train_manifest))

    print(f"[split] base_train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    print(f"[dagger] forced_extra_train={len(extra_files)} repeat={args.extra_repeat}")
    for name, split in [("base_train", train_files), ("val", val_files), ("test", test_files), ("extra_train", extra_files)]:
        print(f"[split] {name}:")
        for item in split:
            print(f"  - {item.name} goals={file_goal_ids(item)}")

    base_train_ds = ManifestV56HistoryDataset(train_files, verbose=not args.quiet)
    extra_train_ds = ManifestV56HistoryDataset(extra_files, verbose=not args.quiet)
    weighted_train_parts = [base_train_ds] + [extra_train_ds] * args.extra_repeat
    train_ds = ConcatDataset(weighted_train_parts)
    val_ds = ManifestV56HistoryDataset(val_files, verbose=not args.quiet)
    test_ds = ManifestV56HistoryDataset(test_files, verbose=not args.quiet)

    stats = {
        "base_train": base_train_ds.label_summary(),
        "extra_train": extra_train_ds.label_summary(),
        "weighted_train": concat_label_summary(weighted_train_parts),
        "val": val_ds.label_summary(),
        "test": test_ds.label_summary(),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "train_files": [str(p) for p in train_files],
        "extra_train_files": [str(p) for p in extra_files],
        "extra_repeat": args.extra_repeat,
        "val_files": [str(p) for p in val_files],
        "test_files": [str(p) for p in test_files],
        "stats": stats,
    }
    (output_dir / "v59_dagger_split_manifest.json").write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    status_path = output_dir / "status_v59_dagger.json"
    if args.dry_run:
        write_status(status_path, status="dry_run_complete", epochs_requested=0)
        return

    run_start = time.time()
    write_status(
        status_path,
        status="running",
        current_epoch=0,
        epochs_requested=args.epochs,
        manifest=str(manifest_path),
        extra_train_manifest=str(args.extra_train_manifest),
        extra_repeat=args.extra_repeat,
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
    best_action = -1.0
    log_path = output_dir / "training_log_v59_dagger.csv"
    log_path.write_text("epoch,train_loss,val_loss,val_acc_action,val_acc_mouse,lr,time\n", encoding="utf-8")

    try:
        for epoch in range(args.epochs):
            model.train()
            start = time.time()
            total_loss = 0.0
            optimizer.zero_grad()
            if args.progress == "epoch":
                print(f"[epoch {epoch + 1}/{args.epochs}] started", flush=True)
            for batch_idx, (frames, actions, buckets, goals) in enumerate(train_loader):
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
                total_loss += float(loss.item()) * GRADIENT_ACCUMULATION_STEPS

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
            save_checkpoint(output_dir / "goal_bc_v59_dagger_last.pt", model, optimizer, epoch + 1, val_metrics, args)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(output_dir / "goal_bc_v59_dagger_best.pt", model, optimizer, epoch + 1, val_metrics, args)
            if val_metrics["acc_action"] > best_action:
                best_action = val_metrics["acc_action"]
                save_checkpoint(output_dir / "goal_bc_v59_dagger_best_action.pt", model, optimizer, epoch + 1, val_metrics, args)

        checkpoint = torch.load(output_dir / "goal_bc_v59_dagger_best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, action_loss_fn, mouse_loss_fn, device)
        (output_dir / "test_metrics_v59_dagger.json").write_text(
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
        write_status(status_path, status="failed", error=str(exc), elapsed_seconds=time.time() - run_start)
        raise


def main():
    parser = argparse.ArgumentParser(description="Train v5.9 DAgger weighted history policy")
    parser.add_argument("--manifest", required=True, help="Base manifest with normal data")
    parser.add_argument("--extra-train-manifest", required=True, help="Forced train correction manifest")
    parser.add_argument("--extra-repeat", type=int, default=5)
    parser.add_argument("--output-dir", default="checkpoints/v59_dagger_history")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-goals", type=int, default=2)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progress", choices=["epoch", "none"], default="epoch")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
