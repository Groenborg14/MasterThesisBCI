import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RawEEGTuner(nn.Module):
    def __init__(self, input_channels, time_window, n_actions):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.encoder(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

    def get_action(self, state):
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()
