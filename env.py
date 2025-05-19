# env.py
import gymnasium as gym
import numpy as np
import torch


def extract_windows(trial: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    n_channels, n_time = trial.shape
    windows = []
    for start in range(0, n_time - window_size + 1, step_size):
        windows.append(trial[:, start:start + window_size])
    return np.stack(windows, axis=0)


class EEGenv(gym.Env):
    def __init__(self, eeg_data: np.ndarray, labels: np.ndarray,
                 window_size: int, step_size: int):
        super(EEGenv, self).__init__()
        self.eeg_data = eeg_data
        self.labels = labels
        self.window_size = window_size
        self.step_size = step_size
        self.n_samples = eeg_data.shape[0]
        self.windows_list = [extract_windows(trial, window_size, step_size)
                             for trial in eeg_data]
        self.n_windows_per_trial = [windows.shape[0] for windows in self.windows_list]
        self.total_windows = sum(self.n_windows_per_trial)

        if self.windows_list:
            n_channels = self.windows_list[0].shape[1]
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(n_channels, window_size),
                dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(0, 0),
                dtype=np.float32
            )
        self.action_space = gym.spaces.Discrete(len(np.unique(labels)))

        self.current_trial = 0
        self.current_window = 0
        self.windows = self.windows_list[self.current_trial]

        self.trial_action_confidences = []

    def reset(self):
        self.trial_action_confidences = []
        self.current_trial = 0
        self.current_window = 0
        if self.windows_list:
            self.windows = self.windows_list[self.current_trial]
            return self.windows[self.current_window]
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action_with_confidence: tuple):
        action, confidence = action_with_confidence
        true_label = int(self.labels[self.current_trial])

        self.trial_action_confidences.append((action, confidence))
        self.current_window += 1
        done = False
        next_state = None
        reward = 0.0  # default reward during window steps

        if self.current_window < self.windows.shape[0]:
            next_state = self.windows[self.current_window]
        else:
            done = True
            # Compute best action by highest confidence
            best_action, _ = max(self.trial_action_confidences, key=lambda x: x[1])
            reward = 10.0 if best_action == true_label else -10.0  # Strong signal at end
            info = {
                'true_label': true_label,
                'predicted_action': best_action,
                'trial_action_confidences': self.trial_action_confidences
            }
            return None, reward, done, info

        info = {
            'true_label': true_label,
            'action': action
        }
        return next_state, reward, done, info
