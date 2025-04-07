import torch
import torch.nn as nn

import torch.optim as optim

from torch import multiprocessing



is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)


class PPOAgent(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(PPOAgent, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # this ouputs logits for each of the 7 classes
        self.policy_head = nn.Linear(64, n_actions)

        # this predicts the value aka the expected return of the current state
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x)
    
    def get_action(self, state):
        # converts logit into a categorical distribution over 7 classes and samples an action from it
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()
    
