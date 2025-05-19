import os
import numpy as np
import mne
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import LeaveOneOut
from scipy.stats import mode
import gymnasium as gym
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import wandb
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from csp import CSP
from csp_nn import CSPTuner
from env import EEGenv  # Assuming your EEGenv is in 'env.py'
from ppo_trainer import ppo_train  # Assuming your ppo_train is in 'ppo_trainer.py'

wandb.login(key="3d885e6576275509407474c4ead6c777ae51d660")  # Replace with your actual Wandb key

class_labels = ["Hands", "Feet"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_raw_and_get_epochs(subjects):
    runs = [6, 10, 14]
    tmin, tmax = -1.0, 4.0
    raw_fnames = eegbci.load_data(subjects, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.annotations.rename(dict(T1="hands", T2="feet"))
    raw.set_eeg_reference(projection=True)
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    epochs = Epochs(
        raw,
        event_id=["hands", "feet"],
        tmin=tmin,
        tmax=tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    epochs_train = epochs.copy().crop(tmin=0, tmax=3)
    labels = epochs.events[:, -1] - 2
    return epochs_train, labels


def evaluate_rl_agent(agent, X_test, y_test, sfreq, class_labels=None, window_size=0.5, step_size=0.1):
    print("Evaluating RL Agent...")
    w_size = int(window_size * sfreq)
    w_step = int(step_size * sfreq)
    env = EEGenv(X_test.astype(np.float32), y_test, window_size=w_size, step_size=w_step)

    y_pred, y_true = [], []
    rewards = []

    for trial in range(env.n_samples):
        env.current_trial = trial
        env.current_window = 0
        env.windows = env.windows_list[trial]
        true_label = int(env.labels[trial])

        highest_confidence = -1.0
        predicted_action_at_highest_confidence = -1
        done = False
        state = torch.tensor(env.windows[env.current_window], dtype=torch.float32)

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # (1, C, T)

            with torch.no_grad():
                logits, _ = agent(state_tensor)
                probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                action = torch.argmax(logits, dim=1).item()
                confidence = probabilities[action]
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                print("Policy output:", probs)

                if confidence > highest_confidence:
                    highest_confidence = confidence
                    predicted_action_at_highest_confidence = action

            env_input = (action, confidence)
            state, reward, done, _ = env.step(env_input)

        final_prediction = predicted_action_at_highest_confidence
        final_reward = 1 if final_prediction == true_label else -1
        rewards.append(final_reward)
        y_pred.append(final_prediction)
        y_true.append(true_label)

    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=range(2))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Highest Confidence)")
    plt.tight_layout()
    plt.savefig("conf_matrix_highest_confidence.png")

    wandb.log({
        "eval/highest_confidence_confusion_matrix": wandb.Image("conf_matrix_highest_confidence.png"),
        "eval/highest_confidence_accuracy": accuracy,
        "eval/highest_confidence_reward_avg": np.mean(rewards)
    })
    plt.close()
    return {"accuracy": accuracy, "confusion_matrix": cm, "y_true": y_true, "y_pred": y_pred}


def train_rl_agent(agent, X_train, y_train, sfreq, window_size=0.5, step_size=0.1):
    print("Training RL Agent...")
    w_size = int(window_size * sfreq)
    w_step = int(step_size * sfreq)
    env = EEGenv(X_train, y_train, window_size=w_size, step_size=w_step)
    print("Observation space:", env.observation_space)
    ppo_train(agent, env)


# === MAIN ===
#wandb.init(project="eeg-ppo-csp", name="leave_one_out_rl_highest_confidence") # Initialize wandb once


subject_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
all_preds, all_labels = [], []
loo = LeaveOneOut()

for fold, (train_idx, test_idx) in enumerate(loo.split(subject_ids)):
    

    train_subjects = subject_ids[train_idx]
    test_subjects = subject_ids[test_idx]

    wandb.init(
    project="eeg-ppo-csp",
    name=f"leave_one_out_fold_test_subject{test_subjects[0]}",
    group="leave_one_out_rl",
    tags=[f"fold_{test_subjects[0]}"],
    reinit=True)


    
   
    

    X_train_band, y_train = load_raw_and_get_epochs(train_subjects)
    X_test_band, y_test = load_raw_and_get_epochs(test_subjects)
    print(f"Fold {test_subjects} - y_test shape:", y_test.shape)
    print(f"Fold {test_subjects} - classes:", np.unique(y_test))

    sfreq = X_train_band.info['sfreq']
    print(f"Fold {test_subjects} - Train label distribution:", np.bincount(y_train))

    csp = CSP(n_components=4)
    csp.fit(X_train_band.get_data(), y_train)
    filters = csp.filters

    X_train_np = X_train_band.get_data().astype(np.float32)
    X_test_np = X_test_band.get_data().astype(np.float32)

    wandb.config.time_window = X_train_np.shape[2]
    wandb.config.n_actions = 2
    wandb.config.ppo_epochs = 10
    wandb.config.rollout_len = 256
    wandb.config.gamma = 0.99
    wandb.config.lr = 2.5e-4

    model = CSPTuner(
        pretrained_filters=filters,
        n_actions=2,
        trainable_csp=False
    ).to(device) # Ensure model is on the correct device

    train_rl_agent(model, X_train_np, y_train, sfreq)
    results = evaluate_rl_agent(model, X_test_np, y_test, sfreq, class_labels)
    wandb.log({
    "fold": test_subjects[0],
    "fold_accuracy": results["accuracy"]
        }, step=fold)

    all_preds.extend(results["y_pred"])
    all_labels.extend(results["y_true"])
    print(f"Fold {test_subjects} Eval Accuracy (Highest Confidence): {results['accuracy']:.4f}")
    wandb.finish()

# === Total Confusion Matrix (Highest Confidence) ===
total_cm = confusion_matrix(all_labels, all_preds, labels=range(2))
plt.figure(figsize=(8, 6))
sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Total Confusion Matrix (Highest Confidence)")
plt.tight_layout()
plt.savefig("total_confusion_matrix_highest_confidence.png")
wandb.log({"total_confusion_matrix_highest_confidence": wandb.Image("total_confusion_matrix_highest_confidence.png")})
plt.show()

wandb.finish()