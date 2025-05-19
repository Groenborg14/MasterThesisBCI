import os
import numpy as np
import mne
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import KFold
import wandb
from sklearn.model_selection import LeaveOneOut
from csp import CSP
from csp_nn import CSPTuner
from ppo_trainer import ppo_train
from env import EEGenv

wandb.login(key="3d885e6576275509407474c4ead6c777ae51d660")

class_labels = ["Left Hand", "Right Hand"]
event_dict = {
    "left_hand": 769,
    "right_hand": 770
}

m=2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_multi_subject_data(subjects, event_dict, tmin=1.5, tmax=5.5):
    from preproc3 import load_bci4_subject_data
    all_alpha, all_beta, all_labels = [], [], []
    for subj in subjects:
        Xa, Xb, y = load_bci4_subject_data(subj, event_dict, tmin, tmax)
        if Xa is not None:
            all_alpha.append(Xa)
            all_beta.append(Xb)
            all_labels.append(y)
    return np.concatenate(all_alpha), np.concatenate(all_beta), np.concatenate(all_labels)

def apply_filters_and_lbp(X, filters):
    # Apply CSP filters: shape -> [n_trials, 2m, time]
    X_csp = np.asarray([np.dot(filters, trial) for trial in X])
    # Log band power
    X_lbp = np.log(np.var(X_csp, axis=2) + 1e-10)  # shape: [n_trials, 2m]
    return X_lbp

def evaluate_rl_agent(agent, X_test, y_test, class_labels=None):
    env = EEGenv(X_test, y_test)
    state = env.reset()
    y_pred, y_true = [], []
    rewards = []
    done = False

    while not done:
        state_tensor = state.unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = agent(state_tensor)
            action = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=-1)
            confidence = probs[0, action].item()

        next_state, _, done, info = env.step(action)
        true_label = info['true_label']

        # Reward shaping logic
        reward = confidence if action == true_label else -confidence
        rewards.append(reward)

        y_pred.append(action)
        y_true.append(true_label)
        state = next_state

    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=range(2))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("conf_matrix.png")

    wandb.log({
        "confusion_matrix": wandb.Image("conf_matrix.png"),
        "test_accuracy": accuracy,
        "eval_reward_avg": np.mean(rewards)
    })
    plt.close()

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred
    }
def test_env(env, agent):
    state = env.reset()
    correct = 0

    for i in range(env.n_samples):
        state_tensor = state.unsqueeze(0).to(device)
        action, _, _ = agent.get_action(state_tensor)
        next_state, reward, done, info = env.step(action)

        print(f"[Step {i}] Action: {action}, True: {info['true_label']}, Reward: {reward}")
        if reward == 1:
            correct += 1

        if done:
            break
        state = next_state

    accuracy = correct / env.n_samples
    print(f"\n✅ Manual Test Accuracy: {accuracy:.2%}")
    return accuracy


def train_rl_agent(agent, X_train, y_train):
    env = EEGenv(X_train, y_train)
    ppo_train(agent, env)

# === MAIN ===
subject_ids = np.array([1, 2,3, 4, 5,6, 7, 8, 9])
#kf = KFold(n_splits=3, shuffle=True, random_state=42)
all_preds, all_labels = [], []
loo = LeaveOneOut()

for train_idx, test_idx in loo.split(subject_ids):
    train_subjects = subject_ids[train_idx]
    test_subjects = subject_ids[test_idx]

    X_train_alpha, X_train_beta, y_train = load_multi_subject_data(train_subjects, event_dict)
    X_test_alpha, X_test_beta, y_test = load_multi_subject_data(test_subjects, event_dict)
    
    print("Train label distribution:", np.bincount(y_train))
    csp_a = CSP(n_components=10)
    csp_b = CSP(n_components=10)
    csp_a.fit(X_train_alpha, y_train)
    csp_b.fit(X_train_beta, y_train)

    filters_alpha = np.concatenate([csp_a.filters[:m], csp_a.filters[-m:]], axis=0)
    filters_beta  = np.concatenate([csp_b.filters[:m], csp_b.filters[-m:]], axis=0)

    X_train_tensor = torch.tensor(np.concatenate([X_train_alpha, X_train_beta], axis=1), dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(np.concatenate([X_test_alpha, X_test_beta], axis=1), dtype=torch.float32).to(device)

    wandb.init(project="eeg-ppo-csp", name=f"alpha_beta_fold_{test_subjects}", config={
        "fold": test_subjects,
        "time_window": X_train_alpha.shape[2],
        "n_actions": 2,
        "ppo_epochs": 10,
        "rollout_len": 256,
        "gamma": 0.99,
        "lr": 2.5e-4
    })

    model = CSPTuner(
        pretrained_alpha=filters_alpha,
        pretrained_beta=filters_beta,
        time_window=X_train_alpha.shape[2],
        n_actions=2,
        trainable_csp=False
    )

    print("y train labels:", y_train[[1,2,3,4,5,6,7,8,9,10]])
    train_rl_agent(model, X_train_tensor, y_train)
    results = evaluate_rl_agent(model, X_test_tensor, y_test, class_labels)

    # === NEW: manual test loop ===
    env_test = EEGenv(X_test_tensor, y_test)
    manual_accuracy = test_env(env_test, model)

    wandb.log({"manual_test_accuracy": manual_accuracy})

    all_preds.extend(results["y_pred"])
    all_labels.extend(results["y_true"])
    print(f"Fold {test_subjects} Eval Accuracy: {results['accuracy']:.4f}")
    wandb.finish()

# === Total Confusion Matrix ===
total_cm = confusion_matrix(all_labels, all_preds, labels=range(2))
plt.figure(figsize=(8, 6))
sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Total Confusion Matrix Across All Folds")
plt.tight_layout()
plt.savefig("total_confusion_matrix.png")
plt.show()

try:
    wandb.init(project="eeg-ppo-csp", name="alpha_beta_total_confusion")
    wandb.log({"total_confusion_matrix": wandb.Image("total_confusion_matrix.png")})
    wandb.finish()
except:
    pass