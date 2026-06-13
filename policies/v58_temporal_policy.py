"""v5.8 temporal GRU policy adapter."""
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import ACTION_NAMES, MOUSE_BUCKET_TO_DX, FrameBatch, Prediction
from models.temporal_gru_bc import GoalConditionedGRU


class V58TemporalPolicy:
    """Run the GRU temporal model over a fixed recent frame window."""

    required_frames = 16

    def __init__(self, model_path, goal_id=0, device="cuda:0", num_goals=2):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.goal_id = goal_id

        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get("model_config", {})
        self.required_frames = int(config.get("seq_length", self.required_frames))
        self.model = GoalConditionedGRU(
            num_goals=int(config.get("num_goals", num_goals)),
            hidden_dim=int(config.get("hidden_dim", 256)),
            gru_layers=int(config.get("gru_layers", 1)),
            dropout=float(config.get("dropout", 0.3)),
        )
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        if config.get("model_kind") == "cached_feature_gru":
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, batch: FrameBatch) -> Prediction:
        if len(batch.frames) != self.required_frames:
            raise ValueError(f"v5.8 temporal policy requires {self.required_frames} frames")

        processed = [self._preprocess_frame(frame) for frame in batch.frames]
        frames_np = np.stack(processed, axis=0)
        frames_tensor = torch.from_numpy(frames_np).unsqueeze(0).to(self.device)
        goal_ids = torch.tensor([self.goal_id], dtype=torch.long).to(self.device)

        with torch.no_grad():
            action_logits, mouse_logits = self.model(frames_tensor, goal_ids)
            action_probs = torch.softmax(action_logits, dim=-1)
            action_conf, action_idx = torch.max(action_probs, dim=-1)
            mouse_probs = torch.softmax(mouse_logits, dim=-1)
            mouse_conf, mouse_bucket = torch.max(mouse_probs, dim=-1)

        action_id = int(action_idx.item())
        bucket = int(mouse_bucket.item())
        return Prediction(
            action_id=action_id,
            action_confidence=float(action_conf.item()),
            mouse_bucket=bucket,
            mouse_confidence=float(mouse_conf.item()),
            action_name=ACTION_NAMES.get(action_id, "unknown"),
            raw_mouse_dx=MOUSE_BUCKET_TO_DX.get(bucket, 0),
        )

    def _preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        return np.transpose(frame_normalized, (2, 0, 1))
