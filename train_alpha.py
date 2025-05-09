# Alpha-only CSP PPO Training Script

from sklearn.model_selection import KFold
import numpy as np
from csp import CSP
from get_data import get_data
from csp_nn import CSPTuner
from ppo_trainer import ppo_train
import torch
from env import EEGenv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA
import wandb
from preproc3 import load_subject_data

wandb.login(key="3d885e6576275509407474c4ead6c777ae51d660")

class_labels = ["Hand Open", "Rest"]
event_dict = {
    #"Elbow_flex": 1,
    "Hand_open":6,
    "Rest": 7
}

def load_multi_subject_data(subjects, runs, event_dict, tmin=0.5, tmax=2.5):
    all_alpha = []
    all_labels = []
    
    for subj in subjects:
        X_alpha, _, y = load_subject_data(subj, runs, event_dict, tmin=tmin, tmax=tmax)
        if X_alpha is not None and y is not None:
            all_alpha.append(X_alpha)
            all_labels.append(y)
    
    if not all_alpha:
        return None, None
    
    X_alpha_all = np.concatenate(all_alpha, axis=0)
    y_all = np.concatenate(all_labels)
    return X_alpha_all, y_all
# Evaluation
def evaluate_rl_agent(agent, X_test, y_test, class_labels=None):
    device = next(agent.parameters()).device
    env = EEGenv(X_test, y_test)
    state = env.reset()

    y_pred, y_true = [], []
    done = False
    while not done:
        state_tensor = state.unsqueeze(0)
        with torch.no_grad():
            logits, _ = agent(state_tensor)
            action = torch.argmax(logits, dim=1).item()

        next_state, reward, done, info = env.step(action)
        y_pred.append(action)
        y_true.append(info['true_label'])
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
        "test_accuracy": accuracy
    })
    plt.close()

    return {"accuracy": accuracy, "confusion_matrix": cm, "y_true": y_true, "y_pred": y_pred}

# Training
def train_rl_agent(agent, X_train, y_train):
    env = EEGenv(X_train, y_train)
    ppo_train(agent, env)

# === MAIN ===
subject_ids = np.array([1, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15])
#subject_ids = np.arange(1,16)
kf = KFold(n_splits=3, shuffle=True, random_state=42)
runs = list(range(1, 11))
all_preds, all_labels = [], []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(subject_ids)):
    train_subjects = subject_ids[train_idx]
    test_subjects = subject_ids[test_idx]

    #X_train_alpha, _, y_train = load_subject_data(train_subjects, runs ,event_dict=event_dict)
    #X_test_alpha, _, y_test = load_subject_data(test_subjects, runs, event_dict=event_dict)

    X_train_alpha, y_train = load_multi_subject_data(train_subjects, runs, event_dict)
    X_test_alpha, y_test = load_multi_subject_data(test_subjects, runs, event_dict)

    # Fix label remapping for binary classification
    y_train = np.where(y_train == 6, 0, 1)  # 6 → 0 (Hand Open), 7 → 1 (Rest)
    y_test  = np.where(y_test == 6, 0, 1)
    # === Fit CSP (Alpha only) ===
    csp_a = CSP(n_components=8)
    csp_a.fit(X_train_alpha, y_train)
    filters_a_array = csp_a.filters[:8]

    # Dummy fit to initialize mean/std
    #dummy_labels = (y_train == 1).astype(int)
    #csp_a.fit(X_train_alpha, dummy_labels)
    #csp_a.filters = filters_a_array

    # CSP transform (optional for PCA or visualization)
    X_csp_alpha = csp_a.transform(X_train_alpha)
    X_train_csp = X_csp_alpha

    # === Raw input tensors (alpha only) ===
    X_train_raw_tensor = torch.tensor(X_train_alpha, dtype=torch.float32)
    X_test_raw_tensor = torch.tensor(X_test_alpha, dtype=torch.float32)

    # === Init model ===
    wandb.init(
        project="eeg-ppo-csp",
        name=f"alpha_only_fold_{fold_idx+1}",
        config={
            "fold": fold_idx + 1,
            "time_window": X_train_alpha.shape[2],
            "n_actions": 2,
            "ppo_epochs": 10,
            "rollout_len": 256,
            "gamma": 0.99,
            "lr": 2.5e-4
        }
    )

    print("Raw input shape:", X_train_raw_tensor.shape)
    print("Alpha CSP shape:", filters_a_array.shape)

    model = CSPTuner(
        pretrained_alpha=filters_a_array,
        time_window=X_train_alpha.shape[2],
        n_actions=2, trainable_csp=False
    )

    print("\n🔍 Running PPO environment test...")

    env = EEGenv(X_train_raw_tensor, y_train)
    state = env.reset()

    step_count = 0
    max_steps = 10
    done = False

    while not done and step_count < max_steps:
        print(f"\nStep {step_count + 1}")
        print("True label:", env.labels[env.current_step])

        action = np.random.choice([0, 1])
        next_state, reward, done, info = env.step(action)

        print("Chosen action:", action)
        print("Reward:", reward)
        print("Done:", done)
        print("Returned true_label from info:", info.get("true_label"))

        state = next_state
        step_count += 1


    train_rl_agent(model, X_train_raw_tensor, y_train)
    eval_results = evaluate_rl_agent(model, X_test_raw_tensor, y_test, class_labels=class_labels)

    all_preds.extend(eval_results["y_pred"])
    all_labels.extend(eval_results["y_true"])
    print(f"Fold {fold_idx + 1} Eval Accuracy: {eval_results['accuracy']:.4f}")
    wandb.finish()

# === Overall Confusion Matrix ===
total_cm = confusion_matrix(all_labels, all_preds, labels=range(2))
plt.figure(figsize=(8, 6))
sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Total Confusion Matrix Across All Folds")
plt.tight_layout()
plt.savefig("total_confusion_matrix.png")
plt.show()

try:
    wandb.init(project="eeg-ppo-csp", name="alpha_only_total_confusion")
    wandb.log({"total_confusion_matrix": wandb.Image("total_confusion_matrix.png")})
    wandb.finish()
except:
    pass
