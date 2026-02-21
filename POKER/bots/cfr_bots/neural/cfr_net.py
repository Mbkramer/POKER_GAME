"""
cfr_net.py
----------
Neural network for CFR self-play training.

Unlike the behavioral cloning model which predicts actions,
this network learns two things simultaneously:
  - Policy head:  probability distribution over actions (Nash policy from CFR)
  - Value head:   expected game value from this state

Training loop:
  1. CFR runs self-play → collects (state_features, nash_policy, state_value)
  2. Neural net trains on those tuples
  3. Repeat → network approximates Nash equilibrium
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),   
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),   
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)


class CFRNet(nn.Module):
    def __init__(self, n_features, n_actions, hidden_dim=128, n_blocks=3, dropout=0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),   
            nn.ReLU(),
            nn.Dropout(dropout),
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)],
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """Returns (policy_logits, value)."""
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """Returns (policy_probs, value) for inference."""
        self.eval()
        logits, value = self(x)
        return F.softmax(logits, dim=-1), value
