import torch
import torch.nn as nn
import torch.nn.functional as F

class CSPTuner(nn.Module):

    def __init__(self, pretrained_w, time_window, n_actions):
        # Pretrained weights
        # time_window: number of time points in each trial
        # n_actions: number of actions of the RL
        super().__init__()


        n_filters, n_channels = pretrained_w.shape


        self.csp = nn.Linear(n_channels, n_filters, bias=False)
        self.csp.weight.data = torch.tensor(pretrained_w, dtype=torch.float32)


        self.policy = nn.Sequential(
            nn.Flatten(),                  # 7 filters × 8 time → 56
            nn.Linear(56, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        
        x_proj = self.csp(x.transpose(1, 2)) # shape: (batch_size, n_filters, time_window) 

        return self.policy(x_proj) # shape: (batch_size, n_actions)
