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
        input_size = pretrained_w.shape[0] * time_window

        self.csp = nn.Linear(2*n_channels, n_filters, bias=False)
        self.csp.weight.data = torch.tensor(pretrained_w, dtype=torch.float32)


        self.shared = nn.Sequential(
        nn.Flatten(),                  # 7 filters × 8 time → 56
        nn.Linear(input_size, 128),
        nn.ReLU()
        )
        
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x_proj = self.csp(x.transpose(1, 2))  # (B, filters, time)
        x = self.shared(x_proj)
        return self.policy_head(x), self.value_head(x)

    def get_action(self, state):
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()
    