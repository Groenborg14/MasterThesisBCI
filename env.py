import gymnasium as gym
import numpy as np



class EEGenv(gym.Env):
    # eeg_data = the preprocessed CSP signal. Labels = true class of the EEG sample
    def __init__(self,eeg_data,labels):
        super(EEGenv, self).__init__()
        self.eeg_data = eeg_data  # shape: (N, channels, time)
        self.labels = labels
        self.current_step = 0
        self.n_samples = eeg_data.shape[0]
        self.n_features = eeg_data.shape[1]  # Number of features (CSP components)

        # Define the observation and action space. obersevation space = CSP EEG feature vector. Action space = 7 classes
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,))
        self.action_space = gym.spaces.Discrete(2) # 7 actions for 7 classes

    def reset(self):
        # this is called at the start of each episode to reset the environment to the first EEG sample
        self.current_step = 0
        return self.eeg_data[self.current_step]
    
    def step(self, action):
        # this compares the agents action to the true label of the EEG sample and returns a reward of +1 if the action is correct and -1 if it is incorrect.
        true_label = self.labels[self.current_step]

        reward = 1 if action == true_label else -1 # reward of +1 for correct action, -1 for incorrect action
        
        # move to the next step
        self.current_step += 1
        # check if the episode is done
        # if the current step is greater than the number of EEG samples, the episode is done
        done = self.current_step >= len(self.eeg_data)
        next_state = self.eeg_data[self.current_step] if not done else None

        info = {
        'true_label': true_label,
        'action': action
        }
        return next_state, reward, done, info



