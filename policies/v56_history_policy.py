"""v5.6 policy adapter using historical frame stacking.

The model architecture is compatible with v5.5, but the temporal semantics are:
  [t-7, t-3, t-1, t] -> action[t]
"""
import numpy as np
import torch

from core.types import ACTION_NAMES, MOUSE_BUCKET_TO_DX, FrameBatch, Prediction
from policies.v55_policy import V55GoalConditionedPolicy


class V56HistoryPolicy(V55GoalConditionedPolicy):
    """Use an 8-frame buffer and select historical frames for v5.6."""

    required_frames = 8
    selected_indices = (0, 4, 6, 7)

    def predict(self, batch: FrameBatch):
        if len(batch.frames) != self.required_frames:
            raise ValueError(f"v5.6 history policy requires {self.required_frames} frames")
        selected = [batch.frames[index] for index in self.selected_indices]
        processed = [self._preprocess_frame(frame) for frame in selected]
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
