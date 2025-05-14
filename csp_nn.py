import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CSPTuner(nn.Module):

    def __init__(self, pretrained_alpha, pretrained_beta, time_window, n_actions, trainable_csp=True):
        super().__init__()

        self.n_filters_alpha, self.n_channels_alpha = pretrained_alpha.shape
        self.n_filters_beta, self.n_channels_beta = pretrained_beta.shape

        self.csp_alpha = nn.Conv1d(self.n_channels_alpha, self.n_filters_alpha, kernel_size=1, bias=False)
        self.csp_beta = nn.Conv1d(self.n_channels_beta, self.n_filters_beta, kernel_size=1, bias=False)

        self.csp_alpha.weight.data = torch.tensor(pretrained_alpha[:, :, None], dtype=torch.float32)
        self.csp_beta.weight.data = torch.tensor(pretrained_beta[:, :, None], dtype=torch.float32)

        self.csp_alpha.weight.requires_grad = trainable_csp
        self.csp_beta.weight.requires_grad = trainable_csp
        print("alfa:",self.n_filters_alpha, self.n_channels_alpha)
        print("beta:",self.n_filters_beta, self.n_channels_beta)
        print(self.n_filters_alpha+self.n_filters_beta)
        
        self.shared = nn.Sequential(
            nn.Linear(self.n_filters_alpha + self.n_filters_beta, 16),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(16),
            nn.Linear(16, 8),
            nn.LeakyReLU()
        )

        self.policy_head = nn.Linear(8, n_actions)
        self.value_head = nn.Linear(8, 1)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x_alpha = x[:, :self.n_channels_alpha, :]
        x_beta = x[:, self.n_channels_alpha:, :]

        proj_alpha = self.csp_alpha(x_alpha)
        proj_beta = self.csp_beta(x_beta)

        var_alpha = torch.var(proj_alpha, dim=2, unbiased=False)
        var_beta = torch.var(proj_beta, dim=2, unbiased=False)

        logvar_alpha = torch.log(var_alpha + 1e-6)
        logvar_beta = torch.log(var_beta + 1e-6)

        x_features = torch.cat([logvar_alpha, logvar_beta], dim=1)
        x = self.shared(x_features)

        logits = self.policy_head(x)
        logits = torch.clamp(logits, -10, 10)
        return logits, self.value_head(x)

    def get_action(self, state):
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()
