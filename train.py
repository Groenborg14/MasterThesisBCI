from sklearn.model_selection import KFold
import numpy as np
from csp import CSP
from get_data import get_data
from csp_nn import CSPTuner
from ppo_trainer import ppo_train
import torch
from env import EEGenv
from torch import multiprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import wandb
from sklearn.decomposition import PCA
from EEG_tuner import RawEEGTuner


from sklearn.preprocessing import StandardScaler

wandb.login(key="3d885e6576275509407474c4ead6c777ae51d660")

class_labels = [
    #"Elbow Flex", #"Elbow Extend", 
    #"Supination",
    #"Pronation", "Hand Close", 
    "Hand Open", 
    "Rest"
]



def train_csp_one_vs_rest_kfold(X, y, n_components=8):
    
    
    filters_per_class = []

    for class_idx in range(2):
        y_binary = (y == class_idx).astype(int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_filters = []
        for train_idx, _ in skf.split(X, y_binary):
            X_fold, y_fold = X[train_idx], y_binary[train_idx]
            csp = CSP(n_components=n_components)
            csp.fit(X_fold, y_fold)
            fold_filters.append(csp.filters[:n_components])

        avg_filters = np.mean(fold_filters, axis=0)
        filters_per_class.append(avg_filters)

    return np.vstack(filters_per_class)  # (7*7, n_channels)

def train_csp_per_band_kfold(X_alpha, X_beta, y, n_components=8):
    filters_alpha = []
    filters_beta = []

    for class_idx in range(2):
        y_binary = (y == class_idx).astype(int)
        #print("y_binary",y_binary.shape)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        fold_filters_alpha = []
        fold_filters_beta = []

        for train_idx, _ in skf.split(X_alpha, y):
            X_alpha_fold = X_alpha[train_idx]
            X_beta_fold = X_beta[train_idx]
            y_fold = y_binary[train_idx]
            #print("X_alpha_fold",X_alpha_fold.shape,"X_beta_fold",X_beta_fold.shape)
            #print(y_fold, "shape",y_fold.shape)

            csp_a = CSP(n_components=n_components)
            csp_a.fit(X_alpha_fold, y_fold)
            fold_filters_alpha.append(csp_a.filters[:n_components])

            csp_b = CSP(n_components=n_components)
            csp_b.fit(X_beta_fold, y_fold)
            fold_filters_beta.append(csp_b.filters[:n_components])

        # Average across folds
        filters_alpha.append(np.mean(fold_filters_alpha, axis=0))
        filters_beta.append(np.mean(fold_filters_beta, axis=0))

    # Stack all classes
    filters_alpha = np.vstack(filters_alpha)  # shape: (7 * n_components, n_channels)
    filters_beta  = np.vstack(filters_beta)

     # Create reusable CSP objects with learned filters
    csp_alpha = CSP(n_components=filters_alpha.shape[0])
    csp_alpha.filters = filters_alpha

    csp_beta = CSP(n_components=filters_beta.shape[0])
    csp_beta.filters = filters_beta

    return filters_alpha, filters_beta
# Evaluation function
def evaluate_rl_agent(agent, X_test, y_test, class_labels=None):
    device = next(agent.parameters()).device
    env = EEGenv(X_test, y_test)
    state = env.reset()

    y_pred = []
    y_true = []

    done = False
    while not done:
      
        state_tensor = state.unsqueeze(0)

        with torch.no_grad():
            logits, _ = agent(state_tensor)
            action = torch.argmax(logits, dim=1).item()

        next_state, reward, done, info = env.step(action)

        y_pred.append(action)
        y_true.append(info['true_label'])  # Ensure this is returned in your env
        state = next_state

    accuracy = np.mean(np.array(y_pred) == np.array(y_true))

    # --- Confusion matrix plotting ---
    cm = confusion_matrix(y_true, y_pred, labels=range(2))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels if class_labels else range(2),
                yticklabels=class_labels if class_labels else range(2))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save and log to wandb
    plt.savefig("conf_matrix.png")
    wandb.log({
        "confusion_matrix": wandb.Image("conf_matrix.png"),
        "test_accuracy": accuracy
    })
    plt.close()

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred
    }

# Train wrapper
def train_rl_agent(agent, X_train, y_train):
    env = EEGenv(X_train, y_train)
    ppo_train(agent, env)

subject_ids = np.arange(1,16)
kf = KFold(n_splits=3, shuffle=True, random_state=42)

runs = list(range(1, 11))
all_preds = []
all_labels = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(subject_ids)):
    train_subjects = subject_ids[train_idx]
    test_subjects = subject_ids[test_idx]

    X_train_alpha,X_train_beta, y_train = get_data(train_subjects, runs)
    X_test_alpha,X_test_beta, y_test = get_data(test_subjects, runs)
    print("X_train_alpha",X_train_alpha.shape,"y train shape",y_train.shape)

    #scale the data
    #scaler_alpha = StandardScaler()
   # X_train_alpha = scaler_alpha.fit_transform(X_train_alpha.reshape(X_train_alpha.shape[0], -1)).reshape(X_train_alpha.shape)
    #X_test_alpha = scaler_alpha.transform(X_test_alpha.reshape(X_test_alpha.shape[0], -1)).reshape(X_test_alpha.shape)

    #scaler_beta = StandardScaler()
   # X_train_beta = scaler_beta.fit_transform(X_train_beta.reshape(X_train_beta.shape[0], -1)).reshape(X_train_beta.shape)
   # X_test_beta = scaler_beta.transform(X_test_beta.reshape(X_test_beta.shape[0], -1)).reshape(X_test_beta.shape)
    #pretrained_w = train_csp_one_vs_rest_kfold(X_train, y_train, n_components=7)
    #filters_a_array, filters_b_array = train_csp_per_band_kfold(X_train_alpha, X_train_beta, y_train)


    csp_a = CSP(n_components=8)
    csp_b = CSP(n_components=8)
    csp_a.fit(X_train_alpha, y_train)
    csp_b.fit(X_train_beta, y_train)
    filters_a_array = csp_a.filters[:8]
    filters_b_array = csp_b.filters[:8]
    pretrained_w = np.concatenate([filters_a_array, filters_b_array], axis=0)

    # Create CSP objects to apply transforms
    csp_a = CSP(n_components=8)
    csp_b = CSP(n_components=8)

    dummy_labels = (y_train == 1).astype(int)
    print(f"Unique classes in dummy labels: {np.unique(dummy_labels)}")
    print("Dummy labels shape",dummy_labels.shape)
    # Dummy fit to set mean/std (won't override filters)
    csp_a.fit(X_train_alpha, dummy_labels)
    csp_b.fit(X_train_beta, dummy_labels)

    # Now inject the actual learned filters
    csp_a.filters = filters_a_array
    csp_b.filters = filters_b_array

    # Apply transforms
    #X_csp_train = np.concatenate([
    #    csp_a.transform(X_train_alpha),
    #    csp_b.transform(X_train_beta)
    #], axis=1)

    X_csp_alpha = csp_a.transform(X_train_alpha)  # shape: (n_trials, n_components)
    X_csp_beta  = csp_b.transform(X_train_beta)
    X_train_csp = np.concatenate([X_csp_alpha, X_csp_beta], axis=1)

    # 🔍 OPTIONAL: Plot histogram of log-variance CSP features
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_csp)

   # plt.scatter(X_pca[y_train == 0, 0], X_pca[y_train == 0, 1], label='Class 0', alpha=0.5)
    #plt.scatter(X_pca[y_train == 1, 0], X_pca[y_train == 1, 1], label='Class 1', alpha=0.5)
    #plt.legend()
    #plt.title("PCA of CSP Features")
    #plt.xlabel("PC1")
   # plt.ylabel("PC2")
    #plt.show()

    #print("📊 Plotting CSP component histograms...")
    #plt.figure(figsize=(12, 6))
    #num_components = min(5, X_train_csp.shape[1])  # Limit to 5 components

   # for i in range(num_components):
    #    plt.hist(X_train_csp[y_train == 0][:, i], bins=30, alpha=0.5, label=f'Class 0 - Comp {i}')
   #     plt.hist(X_train_csp[y_train == 1][:, i], bins=30, alpha=0.5, label=f'Class 1 - Comp {i}')
    #    plt.title(f"CSP Component {i} - Log-Variance Distribution")
   #     plt.xlabel("Feature Value")
    #    plt.ylabel("Frequency")
    #    plt.legend()
    #    plt.tight_layout()
    #    plt.show()
    
    X_train_raw = np.concatenate([X_train_alpha, X_train_beta], axis=1)  # Shape: (n_trials, 2*n_channels, time_window)
    X_test_raw  = np.concatenate([X_test_alpha, X_test_beta], axis=1)

    X_train_raw_tensor = torch.tensor(X_train_raw, dtype=torch.float32)
    X_test_raw_tensor = torch.tensor(X_test_raw, dtype=torch.float32)

    X_csp_test = np.concatenate([
        csp_a.transform(X_test_alpha),
        csp_b.transform(X_test_beta)
    ], axis=1)
    
    time_window = X_train_alpha.shape[2]
    n_actions = 2

    wandb.init(
        project="eeg-ppo-csp",
        name=f"fold_{fold_idx+1}",
        config={
            "fold": fold_idx + 1,
            #"n_filters": X_csp_train.shape[0],
            "time_window": time_window,
            "n_actions": n_actions,
            "ppo_epochs": 10,
            "rollout_len": 256,
            "gamma": 0.99,
            "lr": 2.5e-4
        }
    )
    # view the class distribution in the training and test sets
    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print("Class distribution (train):", class_distribution)

    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print("Class distribution (test):", dict(zip(unique_test, counts_test)))


    #input_dim = X_csp_train.shape[1]  # typically 14 (7 filters per band × 2 bands)
    model = CSPTuner(pretrained_alpha=filters_a_array,
                    pretrained_beta=filters_b_array,
                    time_window=X_train_alpha.shape[2],
                    n_actions=2)
    #model = RawEEGTuner(input_channels = 30,time_window= X_test_alpha.shape[2],n_actions=2)
    print("Raw input shape:", X_train_raw_tensor.shape)        # e.g., (1200, 30, 1025)
    print("Alpha CSP shape:", filters_a_array.shape)           # e.g., (16, 15)
    print("Beta CSP shape:", filters_b_array.shape)            # e.g., (16, 15)

    
    train_rl_agent(model, X_train_raw_tensor, y_train)
    #train_rl_agent(model, X_train_raw_tensor, y_train)
    eval_results = evaluate_rl_agent(model, X_test_raw_tensor, y_test, class_labels=class_labels)
    all_preds.extend(eval_results["y_pred"])
    all_labels.extend(eval_results["y_true"])

    print(f"Fold {fold_idx + 1} Eval Accuracy: {eval_results['accuracy']:.4f}")
    wandb.finish()

# --- Total Confusion Matrix Across All Folds ---
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

# Optional W&B log
try:
    wandb.init(project="eeg-ppo-csp", name="total_confusion_matrix")
    wandb.log({"total_confusion_matrix": wandb.Image("total_confusion_matrix.png")})
    wandb.finish()
except:
    pass