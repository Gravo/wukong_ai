#!/usr/bin/env python3
"""Train a fast v5.8 temporal GRU baseline using cached ResNet frame features."""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import torchvision.models as tv_models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.temporal_gru_bc import FeatureGoalConditionedGRU
from training.goal_conditioned_bc_v55_optimized import FocalLoss
from training.train_v56_history_stack_clean import (
    action_from_dx,
    classify_bucket,
    file_goal_ids,
    load_manifest,
    split_files,
    write_split_manifest,
    write_status,
)


def build_feature_encoder(device):
    resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    encoder = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def encode_frames(frames_np, encoder, device, batch_size):
    features = []
    with torch.no_grad():
        for start in range(0, len(frames_np), batch_size):
            batch = torch.from_numpy(frames_np[start : start + batch_size]).float() / 255.0
            batch = batch.permute(0, 3, 1, 2).to(device)
            feat = encoder(batch).reshape(batch.shape[0], 512).cpu().numpy().astype(np.float32)
            features.append(feat)
    return np.concatenate(features, axis=0)


def ensure_feature_cache(files, cache_dir: Path, encoder, device, batch_size):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = {}
    for path in tqdm(files, desc="feature-cache"):
        cache_path = cache_dir / f"{path.stem}.npz"
        if not cache_path.exists():
            with h5py.File(path, "r") as hf:
                frames = hf["frames"][:]
                mouse_dx = hf["mouse_dx"][:].astype(np.float32)
                goal_ids = hf["goal_ids"][:] if "goal_ids" in hf else np.zeros(len(frames), dtype=np.int64)
            features = encode_frames(frames, encoder, device, batch_size)
            np.savez_compressed(
                cache_path,
                features=features,
                mouse_dx=mouse_dx,
                goal_ids=goal_ids.astype(np.int64),
                source=str(path),
            )
        cache_paths[path] = cache_path
    return cache_paths


class CachedFeatureSequenceDataset(Dataset):
    def __init__(self, files, cache_paths, seq_length):
        self.seq_length = seq_length
        self.samples = []
        self.features_cache = {}
        self.mouse_dx_cache = {}
        self.goal_ids_cache = {}
        for file_idx, path in enumerate(files):
            payload = np.load(cache_paths[path])
            features = payload["features"]
            self.features_cache[file_idx] = features
            self.mouse_dx_cache[file_idx] = payload["mouse_dx"]
            self.goal_ids_cache[file_idx] = payload["goal_ids"]
            for frame_idx in range(seq_length - 1, len(features)):
                self.samples.append((file_idx, frame_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, frame_idx = self.samples[idx]
        start = frame_idx - self.seq_length + 1
        features = torch.from_numpy(self.features_cache[file_idx][start : frame_idx + 1]).float()
        mouse_dx = float(self.mouse_dx_cache[file_idx][frame_idx])
        goal_id = int(self.goal_ids_cache[file_idx][frame_idx])
        return (
            features,
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=shuffle)


def evaluate(model, loader, action_loss_fn, mouse_loss_fn, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct_action = 0
    correct_mouse = 0
    with torch.no_grad():
        for features, actions, buckets, goals in loader:
            features = features.to(device)
            actions = actions.to(device)
            buckets = buckets.to(device)
            goals = goals.to(device)
            action_logits, mouse_logits = model(features, goals)
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


def save_checkpoint(path, model, optimizer, epoch, val_metrics, manifest_path, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "manifest": str(manifest_path),
            "model_config": {
                "model_kind": "cached_feature_gru",
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path = Path(args.manifest)
    files = load_manifest(manifest_path)
    train_files, val_files, test_files = split_files(files, args.val_ratio, args.test_ratio, args.seed)
    print(f"[split] train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    for name, split in [("train", train_files), ("val", val_files), ("test", test_files)]:
        print(f"[split] {name}:")
        for item in split:
            print(f"  - {item.name} goals={file_goal_ids(item)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder = build_feature_encoder(device)
    cache_paths = ensure_feature_cache(files, Path(args.cache_dir), encoder, device, args.encode_batch_size)

    train_ds = CachedFeatureSequenceDataset(train_files, cache_paths, args.seq_length)
    val_ds = CachedFeatureSequenceDataset(val_files, cache_paths, args.seq_length)
    test_ds = CachedFeatureSequenceDataset(test_files, cache_paths, args.seq_length)
    stats = {
        "train": train_ds.label_summary(),
        "val": val_ds.label_summary(),
        "test": test_ds.label_summary(),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifest(output_dir / "v58_temporal_cached_split_manifest.json", train_files, val_files, test_files, stats)
    status_path = output_dir / "status_v58_temporal_cached.json"
    if args.dry_run:
        write_status(status_path, status="dry_run_complete", epochs_requested=0)
        return

    model = FeatureGoalConditionedGRU(
        num_goals=args.num_goals,
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
    ).to(device)
    action_loss_fn = FocalLoss(weight=torch.tensor([1.0, 2.0, 2.0], device=device), gamma=2.0)
    mouse_loss_fn = FocalLoss(weight=torch.tensor([1.5, 2.0, 2.0, 1.0, 2.0, 2.0, 1.5], device=device), gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False)

    log_path = output_dir / "training_log_v58_temporal_cached.csv"
    log_path.write_text("epoch,train_loss,val_loss,val_acc_action,val_acc_mouse,lr,time\n", encoding="utf-8")
    best_val = float("inf")
    best_action = -1.0
    run_start = time.time()
    write_status(status_path, status="running", current_epoch=0, epochs_requested=args.epochs)
    try:
        for epoch in range(args.epochs):
            model.train()
            start = time.time()
            total_loss = 0.0
            if args.progress == "epoch":
                print(f"[epoch {epoch + 1}/{args.epochs}] started", flush=True)
            for features, actions, buckets, goals in train_loader:
                features = features.to(device)
                actions = actions.to(device)
                buckets = buckets.to(device)
                goals = goals.to(device)
                optimizer.zero_grad()
                action_logits, mouse_logits = model(features, goals)
                loss = action_loss_fn(action_logits, actions) + mouse_loss_fn(mouse_logits, buckets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item())
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
            save_checkpoint(output_dir / "goal_bc_v58_temporal_cached_last.pt", model, optimizer, epoch + 1, val_metrics, manifest_path, args)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(output_dir / "goal_bc_v58_temporal_cached_best.pt", model, optimizer, epoch + 1, val_metrics, manifest_path, args)
            if val_metrics["acc_action"] > best_action:
                best_action = val_metrics["acc_action"]
                save_checkpoint(output_dir / "goal_bc_v58_temporal_cached_best_action.pt", model, optimizer, epoch + 1, val_metrics, manifest_path, args)

        checkpoint = torch.load(output_dir / "goal_bc_v58_temporal_cached_best_action.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, action_loss_fn, mouse_loss_fn, device)
        (output_dir / "test_metrics_v58_temporal_cached.json").write_text(
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
    parser = argparse.ArgumentParser(description="Train cached-feature v5.8 GRU baseline")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="checkpoints/v58_temporal_gru_cached_baseline")
    parser.add_argument("--cache-dir", default="feature_cache/v58_resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--seq-length", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-goals", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress", choices=["epoch", "none"], default="epoch")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
