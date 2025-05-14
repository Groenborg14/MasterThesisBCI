import gymnasium as gym
import numpy as np
import torch


class EEGenv(gym.Env):
    # eeg_data = the preprocessed CSP signal. Labels = true class of the EEG sample
    def __init__(self,eeg_data,labels):
        super(EEGenv, self).__init__()
        self.eeg_data = eeg_data  # shape: (N, channels, time)
        self.labels = labels
        assert self.eeg_data.shape[0] == self.labels.shape[0], "Mismatch in number of samples and labels"
        self.current_step = 0
        self.n_samples = eeg_data.shape[0]
        self.n_channels = eeg_data.shape[1]  # Number of features (CSP components)
        self.n_time_window = eeg_data.shape[2]  # Number of time points in each trial

        # Define the observation and action space. obersevation space = CSP EEG feature vector. Action space = 7 classes
        self.observation_space = gym.spaces.Box(low=-np.inf, 
                                                high=np.inf, 
                                                shape=(self.n_channels,self.n_time_window),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2) # 7 actions for 7 classes

    def reset(self):
        # this is called at the start of each episode to reset the environment to the first EEG sample
        self.current_step = 0
        if self.n_samples == 0:
            raise ValueError("EEGenv received empty eeg_data — cannot reset.")
        return self.eeg_data[self.current_step]
    
    def step(self, action):
        # this compares the agents action to the true label of the EEG sample and returns a reward of +1 if the action is correct and -1 if it is incorrect.
        true_label = self.labels[self.current_step]
        
       # if epoch is not None and epoch < 10:
        #    reward = 1 if action == true_label else -1
       # elif logits is not None:
        #    probs = torch.softmax(logits, dim=-1)
        #    confidence = probs[0, action].item()
        #    if action == true_label:
        #        reward = 0.5 + 0.5 * (1 - abs(confidence - 0.5) * 2)
        #    else:
        #        reward = -confidence
        #else:
        #    reward = 1 if action == true_label else -1  # <-- Fallback default
        # Default to ±1 reward
        reward = 1 if action == true_label else 0

        # Optional reward shaping using logits
        

        self.current_step += 1
        done = self.current_step >= len(self.eeg_data)
        next_state = self.eeg_data[self.current_step] if not done else None

        info = {
            'true_label': true_label,
            'action': action
        }
        return next_state, reward, done, info



