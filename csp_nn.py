import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CSPTuner(nn.Module):

    def __init__(self, pretrained_w, time_window, n_actions):
        # Pretrained weights
        # time_window: number of time points in each trial
        # n_actions: number of actions of the RL
        super().__init__()


        n_filters, n_channels = pretrained_w.shape
        

        self.csp = nn.Linear(n_channels, n_filters, bias=False)
        self.csp.weight.data = torch.tensor(pretrained_w, dtype=torch.float32)

        #input_size = n_filters * time_window

        self.shared = nn.Sequential(
        #nn.Flatten(),                  # 7 filters × 8 time → 56
        nn.Linear(n_filters, 128),
        nn.LeakyReLU()
        )
        
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Input contains NaN or inf values!")
       # print(f"Input shape: {x.shape}")
        x_proj = self.csp(x)
        if torch.isnan(x_proj).any() or torch.isinf(x_proj).any():
            print("CSP output contains NaN or inf values!")
        x = self.shared(x_proj)
        logits = self.policy_head(x)
        logits = torch.clamp(logits, -10, 10)
        #print("Logits stats — min:", logits.min().item(), "max:", logits.max().item(), "mean:", logits.mean().item())
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Logits contain NaN or inf values!")
        return logits, self.value_head(x)

    def get_action(self, state):
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()
    