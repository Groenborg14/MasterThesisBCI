# csp_nn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CSPTuner(nn.Module):
    def __init__(self, pretrained_filters, n_actions, trainable_csp=True):
        super().__init__()

        self.n_filters, self.n_channels = pretrained_filters.shape
        self.csp = nn.Conv1d(self.n_channels, self.n_filters, kernel_size=1, bias=False)
        self.csp.weight.data = torch.tensor(pretrained_filters[:, :, None], dtype=torch.float32)
        self.csp.weight.requires_grad = trainable_csp
        print("filters size:", self.n_filters)
        self.shared = nn.Sequential(
            nn.Linear(self.n_filters, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.LeakyReLU()
        )
        
        self.policy_head = nn.Linear(64, n_actions)
        self.value_head = nn.Linear(64, 1)

        nn.init.kaiming_normal_(self.policy_head.weight, nonlinearity='leaky_relu')
        nn.init.zeros_(self.policy_head.bias)
        nn.init.kaiming_normal_(self.value_head.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.value_head.bias, 1.0)

    def forward(self, x):  # x: (batch, channels, time)
        if x.ndim == 4:
            B, W, C, T = x.shape
            x = x.view(B * W, C, T)
        elif x.ndim == 3:
            B, C, T = x.shape
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")

        proj = self.csp(x)  # shape: (B, n_filters, T)
        logvar = torch.log(torch.var(proj, dim=2) + 1e-6)
        x_features = logvar  # shape: (B, n_filters)

        shared_out = self.shared(x_features)
        logits = self.policy_head(shared_out)
        values = self.value_head(shared_out)
        return logits, values

    def get_action(self, state):
        logits, _ = self.forward(state)  # Get raw logits
        probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities
        dist = torch.distributions.Categorical(probs)  # Use probs instead of logits to match softmax
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Extract softmax confidence of the selected action
        confidence = probs[0, action.item()].item()

        return action.item(), log_prob, entropy, confidence
