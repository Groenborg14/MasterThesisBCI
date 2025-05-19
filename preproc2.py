import mne
import numpy as np
from filter import filter_class
from csp import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.covariance import LedoitWolf
from scipy.linalg import logm

def safe_logm(cov, eps=1e-6, complex_tol=1e-8):
    """Compute matrix logarithm safely, handling singular and complex cases."""
    # Add small regularization to the diagonal
    cov += np.eye(cov.shape[0]) * eps

    try:
        log_cov = logm(cov)

        # Check if imaginary part is negligible
        if np.max(np.abs(log_cov.imag)) < complex_tol:
            return log_cov.real
        else:
            raise ValueError("Matrix logarithm resulted in significant complex values.")
    except Exception as e:
        print(f"Warning: logm failed for a matrix. Replacing with zeros. Error: {e}")
        return np.zeros_like(cov)

def compute_logcov_features(X):
    feats = []
    for trial in X:
        cov = LedoitWolf().fit(trial.T).covariance_
        log_cov = safe_logm(cov)
        feats.append(log_cov[np.triu_indices_from(log_cov)])
    return np.array(feats)

# --- CONFIG ---
subjects = [13, 14]
runs = list(range(1, 11))

# Pick two distinct classes
event_dict = {
    "Elbow_flex": 1,
    "Hand_open": 6,
    "Rest": 7
}
class_pair = ("Hand_open", "Rest")
band_alpha = (8, 12)
band_beta  = (12, 30)
epoch_window = (0.5, 2.5)

# --- LOAD AND PREPROCESS ---
def load_raw(subjects, runs):
    raw_list = []
    for subj in subjects:
        base = f"c:/Users/Christian/Desktop/Master Thesis/MI_data/motorimagination_subject{subj}"
        for run in runs:
            path = f"{base}_run{run}.gdf"
            if os.path.exists(path):
                raw = mne.io.read_raw_gdf(path, preload=True,include=(
                "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                "P3", "P1", "Pz", "P2", "P4", "F1", "Fz", "F2"
            ))
                raw.set_eeg_reference('average')
                raw.apply_function(lambda x: x * 1e-6)
                raw_list.append(raw)
    mne.channels.equalize_channels(raw_list)
    return mne.concatenate_raws(raw_list)

def bandpass_and_epoch(raw, band, tmin, tmax):
    data = filter_class(*band).filter_data(raw._data)
    for ch in range(data.shape[0]):
        ch_data = data[ch]
        mask = np.isnan(ch_data) | np.isinf(ch_data)
        ch_data[mask] = np.nanmean(ch_data[~mask]) if np.any(~mask) else 0
    raw_band = mne.io.RawArray(data, raw.info)
    events, _ = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw_band, events, event_id=event_dict, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True)
    return epochs

def prepare_data(epochs, label1, label2):
    X = epochs.get_data()
    y = epochs.events[:, 2]
    keep = (y == event_dict[label1]) | (y == event_dict[label2])
    X = X[keep]
    y = y[keep]
    y_bin = (y == event_dict[label1]).astype(int)
    mask = ~np.isnan(X).any(axis=(1, 2)) & ~np.isinf(X).any(axis=(1, 2))
    return X[mask], y_bin[mask]

# --- CSP + Logistic Regression ---
def run_csp_logreg(X_alpha, X_beta, y):
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(y)), y, test_size=0.2, stratify=y, random_state=42)

    csp_a = CSP()
    csp_b = CSP()
    X_train_a = csp_a.fit_transform(X_alpha[X_train_idx], y_train)
    X_train_b = csp_b.fit_transform(X_beta[X_train_idx], y_train)
    X_train = np.concatenate((X_train_a, X_train_b), axis=1)

    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train, y_train)

    X_test_a = csp_a.transform(X_alpha[X_test_idx])
    X_test_b = csp_b.transform(X_beta[X_test_idx])
    X_test = np.concatenate((X_test_a, X_test_b), axis=1)

    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# --- Riemannian Classifier ---
def run_manual_riemannian(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Extract log-cov features
    feats_train = compute_logcov_features(X_train)
    feats_test = compute_logcov_features(X_test)

    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(feats_train, y_train)
    y_pred = clf.predict(feats_test)
    return accuracy_score(y_test, y_pred)

# --- MAIN SCRIPT ---
raw = load_raw(subjects, runs)
tmin, tmax = epoch_window

epochs_alpha = bandpass_and_epoch(raw, band_alpha, tmin, tmax)
epochs_beta  = bandpass_and_epoch(raw, band_beta, tmin, tmax)

X_alpha, y = prepare_data(epochs_alpha, *class_pair)
X_beta, _  = prepare_data(epochs_beta, *class_pair)

print(f"\nTesting class pair: {class_pair[0]} vs {class_pair[1]}")
print(f"Trials kept: {len(y)}")

X_combined = np.concatenate((X_alpha, X_beta), axis=1)
acc_manual_riem = run_manual_riemannian(X_combined, y)
print(f"✅ Manual Riemannian (log-cov) Accuracy: {acc_manual_riem:.2f}")
