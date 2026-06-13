"""v5.5 Goal-Conditioned BC policy adapter."""
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import ACTION_NAMES, MOUSE_BUCKET_TO_DX, FrameBatch, Prediction
from training.goal_conditioned_bc_v55_optimized import GoalConditionedBC_v55


class V55GoalConditionedPolicy:
    """Wrap the v5.5 model behind the PolicyPort contract."""

    required_frames = 4

    def __init__(self, model_path, goal_id=0, device="cuda:0", num_goals=2):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.goal_id = goal_id
        self.model = GoalConditionedBC_v55(num_goals=num_goals, freeze_backbone=False)

        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, batch: FrameBatch) -> Prediction:
        if len(batch.frames) != self.required_frames:
            raise ValueError(f"v5.5 policy requires {self.required_frames} frames")

        processed = [self._preprocess_frame(frame) for frame in batch.frames]
        frames_np = np.concatenate(processed, axis=0)
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

