"""Temporal goal-conditioned behavior cloning model with a GRU head."""
import torch
import torch.nn as nn
import torchvision.models as models


NUM_CLASSES_ACTION = 3
NUM_CLASSES_MOUSE = 7


class GoalConditionedGRU(nn.Module):
    """Encode each frame with ResNet18, then model short history with a GRU."""

    def __init__(
        self,
        num_goals=2,
        hidden_dim=256,
        gru_layers=1,
        dropout=0.3,
        freeze_backbone=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.goal_embedding = nn.Embedding(num_goals, 512)
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES_ACTION),
        )
        self.mouse_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES_MOUSE),
        )

    def forward(self, frames, goal_ids):
        """Run model.

        Args:
            frames: (B, T, 3, 224, 224)
            goal_ids: (B,) or (B, T)
        """
        batch_size, seq_len = frames.shape[:2]
        flat = frames.reshape(batch_size * seq_len, 3, 224, 224)
        features = self.backbone(flat).reshape(batch_size, seq_len, 512)

        if goal_ids.dim() == 2:
            goal_ids = goal_ids[:, -1]
        goal_emb = self.goal_embedding(goal_ids).unsqueeze(1)
        features = features + goal_emb

        gru_out, _ = self.gru(features)
        last_hidden = self.dropout(gru_out[:, -1, :])
        return self.action_head(last_hidden), self.mouse_head(last_hidden)


class FeatureGoalConditionedGRU(nn.Module):
    """GRU classifier that consumes precomputed 512-d frame features."""

    def __init__(self, num_goals=2, hidden_dim=256, gru_layers=1, dropout=0.3):
        super().__init__()
        self.goal_embedding = nn.Embedding(num_goals, 512)
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES_ACTION),
        )
        self.mouse_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES_MOUSE),
        )

    def forward(self, features, goal_ids):
        if goal_ids.dim() == 2:
            goal_ids = goal_ids[:, -1]
        goal_emb = self.goal_embedding(goal_ids).unsqueeze(1)
        gru_out, _ = self.gru(features + goal_emb)
        last_hidden = self.dropout(gru_out[:, -1, :])
        return self.action_head(last_hidden), self.mouse_head(last_hidden)
