import os
import numpy as np
import mne
from sklearn.model_selection import train_test_split, LeaveOneOut, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from csp import CSP

def drop_bad_epochs(epochs):
    data = epochs.get_data()
    mask = ~np.isnan(data).any(axis=(1, 2)) & ~np.isinf(data).any(axis=(1, 2))
    return epochs[mask]

def interpolate_missing(raw):
    raw_data = raw.get_data()
    if np.isnan(raw_data).any():
        print("🔧 Interpolating missing values (NaNs) in raw data")
        for ch in range(raw_data.shape[0]):
            signal = raw_data[ch]
            nan_mask = np.isnan(signal)
            if nan_mask.any():
                not_nan = ~nan_mask
                raw_data[ch, nan_mask] = np.interp(
                    np.flatnonzero(nan_mask),
                    np.flatnonzero(not_nan),
                    signal[not_nan]
                )
        raw._data = raw_data

def bandpass_filter(raw, l_freq, h_freq):
    filtered = raw.copy().filter(l_freq, h_freq, method='iir')
    return filtered


def load_bci4_subject_data(subject_id, event_dict, tmin=-5, tmax=10):
    base_path = r"C:\Users\Christian\Desktop\Master Thesis\BCI4"
    filename = os.path.join(base_path, f"A0{subject_id}T.gdf")
    if not os.path.exists(filename):
        print(f"⚠️ File not found: {filename}")
        return None, None, None

    raw = mne.io.read_raw_gdf(filename, preload=True, exclude=("EOG-central","EOG-left", "EOG-right", "EEG-5"))
    #interpolate_missing(raw)
    raw.set_eeg_reference('average')
    

    #raw._data *= 1e6  # Convert to microvolts

    # Reject bad channels based on variance
    data = raw.get_data()
    variances = np.var(data, axis=1)
    low_thresh = np.percentile(variances, 2)
    high_thresh = np.percentile(variances, 98)
    bad_idx = np.where((variances < low_thresh) | (variances > high_thresh))[0]
    bad_channels = [raw.ch_names[i] for i in bad_idx]
    if bad_channels:
        print(f"🚫 Rejected channels (subject {subject_id}): {bad_channels}")
        raw.info['bads'] = bad_channels
        try:
            raw.interpolate_bads(origin=(0, 0, 0.04))
        except Exception as e:
            print(f"⚠️ Could not interpolate bad channels for subject {subject_id}: {e}")

    #filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
    #ica = mne.preprocessing.ICA(n_components=10, random_state=97, method='fastica')
    #ica.fit(filt_raw)
    #raw = ica.apply(raw)

    #raw_alpha = bandpass_filter(raw, 8, 34)
    
    #raw_beta = bandpass_filter(raw, 14, 34)

    events, event_id_map = mne.events_from_annotations(raw)
    print("📌 Event descriptions available:", list(event_id_map.keys()))

    # Exclude rejected trials and eye movements
    exclude_ids = [1023, 1072]
    events = np.array([e for e in events if e[2] in [ event_id_map["769"], event_id_map["770"]] ])

    print("events:",events)
    
    event_dict_mapped = {
        "left_hand": event_id_map["769"],
        "right_hand": event_id_map["770"]
    }

    epochs_alpha = mne.Epochs(
        raw, events, event_id=event_dict_mapped, tmin=tmin, tmax=tmax,
        baseline=None, preload=True
    )
    epochs_beta = mne.Epochs(
        raw, events, event_id=event_dict_mapped, tmin=tmin, tmax=tmax,
        baseline=None, preload=True
    )
    #epochs_alpha.plot()

    #epochs_alpha.plot(events=events, event_id=event_dict_mapped, show=True,block=True, n_epochs=1)

    epochs_alpha.filter(8, 34, method='iir')
    epochs_alpha = epochs_alpha.crop(3,6)

    
    data_alpha = epochs_alpha.get_data()
    data_beta = epochs_beta.get_data()
    #time = np.arange(data_alpha.shape[-1])/data_alpha.shape[-1] * (tmax - tmin) + tmin

    #plt.plot(time,data_alpha[0, 0, :])
    #plt.show()


    labels = epochs_alpha.events[:, 2]
    good_idx = ~np.isnan(data_alpha).any(axis=(1, 2)) & ~np.isinf(data_alpha).any(axis=(1, 2))

    label_map = {
        event_dict_mapped["left_hand"]: 0,
        event_dict_mapped["right_hand"]: 1
    }
    labels = np.vectorize(label_map.get)(labels)

    return data_alpha[good_idx], data_beta[good_idx], labels[good_idx]

#def train_and_eval_lda(X_alpha, X_beta, y):
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(y)), y, test_size=0.1, stratify=y, random_state=42)

    csp_alpha = CSP(n_components=12)
    csp_beta = CSP(n_components=12)

    X_train_alpha = X_alpha[X_train_idx]
    X_test_alpha = X_alpha[X_test_idx]
    X_train_beta = X_beta[X_train_idx]
    X_test_beta = X_beta[X_test_idx]
    


    X_csp_alpha_train = csp_alpha.fit_transform(X_train_alpha, y_train)
    X_csp_beta_train = csp_beta.fit_transform(X_train_beta, y_train)

    
    X_csp_train = np.concatenate([X_csp_alpha_train, X_csp_beta_train], axis=1)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_csp_train, y_train)

    X_csp_alpha_test = csp_alpha.transform(X_test_alpha)
    X_csp_beta_test = csp_beta.transform(X_test_beta)
    X_csp_test = np.concatenate([X_csp_alpha_test, X_csp_beta_test], axis=1)

    y_pred = lda.predict(X_csp_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    return accuracy, y_test, y_pred

def plot_csp_features(X_features, y, title="CSP + Log Band Power Features"):
    """
    Plot first two CSP features in 2D for visualizing class separability.
    
    Parameters:
        X_features: np.ndarray of shape (n_samples, n_features)
            The feature matrix after log-variance (LBP) computation.
        y: np.ndarray of shape (n_samples,)
            Class labels (e.g., 0 or 1)
    """
    plt.figure(figsize=(7, 5))

    for label in np.unique(y):
        idx = y == label
        plt.scatter(X_features[idx, 0], X_features[idx, 1],
                    label=f'Class {label}', alpha=0.7)

    plt.title(title)
    plt.xlabel("CSP Feature 1 (log-var)")
    plt.ylabel("CSP Feature 2 (log-var)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_and_eval_lda(X_alpha, X_beta, y):

    
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(y)), y, test_size=0.1, stratify=y, random_state=42)

    X_train_alpha = X_alpha[X_train_idx]
    X_test_alpha = X_alpha[X_test_idx]
    X_train_beta = X_beta[X_train_idx]
    X_test_beta = X_beta[X_test_idx]

    best_accuracy = 0
    best_m = None
    best_y_pred = None

    for m in [1, 2, 3, 4, 5,6]:
        total_components = 12
        csp_alpha = CSP(n_components=total_components)
        csp_beta = CSP(n_components=total_components)

        X_csp_alpha_train_full = csp_alpha.fit_transform(X_train_alpha, y_train)
        X_csp_beta_train_full = csp_beta.fit_transform(X_train_beta, y_train)

        X_csp_alpha_train = np.concatenate([
            X_csp_alpha_train_full[:, :m],
            X_csp_alpha_train_full[:, -m:]
        ], axis=1)
        X_csp_beta_train = np.concatenate([
            X_csp_beta_train_full[:, :m],
            X_csp_beta_train_full[:, -m:]
        ], axis=1)

        alpha_logvar = np.log(np.var(X_csp_alpha_train, axis=1, keepdims=True))
        beta_logvar = np.log(np.var(X_csp_beta_train, axis=1, keepdims=True))
        X_features_train = np.concatenate([alpha_logvar, beta_logvar], axis=1)

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_features_train, y_train)

        X_csp_alpha_test_full = csp_alpha.transform(X_test_alpha)
        X_csp_beta_test_full = csp_beta.transform(X_test_beta)

        X_csp_alpha_test = np.concatenate([
            X_csp_alpha_test_full[:, :m],
            X_csp_alpha_test_full[:, -m:]
        ], axis=1)
        X_csp_beta_test = np.concatenate([
            X_csp_beta_test_full[:, :m],
            X_csp_beta_test_full[:, -m:]
        ], axis=1)

        alpha_logvar_test = np.log(np.var(X_csp_alpha_test, axis=1, keepdims=True))
        beta_logvar_test = np.log(np.var(X_csp_beta_test, axis=1, keepdims=True))
        X_features_test = np.concatenate([alpha_logvar_test, beta_logvar_test], axis=1)

        y_pred = lda.predict(X_features_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"m = {m} → Accuracy: {acc:.2f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_m = m
            best_y_pred = y_pred

    print(f"\n✅ Best m = {best_m} with accuracy = {best_accuracy:.2f}")
    print(classification_report(y_test, best_y_pred))

    return best_accuracy, y_test, best_y_pred, best_m

def kfold_search_best_m_csp_lbp_cv(X_alpha, X_beta, y, m_range=range(11,12), k=6, n_components=10):
    """
    Perform k-fold CV and search for best `m` (number of CSP filters to keep from each side).

    Parameters:
        X_alpha, X_beta : np.ndarray
            EEG data in alpha and beta bands. Shape: (trials, channels, time)
        y : np.ndarray
            Class labels
        m_range : iterable
            Range of m values to test (e.g., range(1, 6))
        k : int
            Number of folds for cross-validation
        n_components : int
            Total number of CSP components (should be > 2 * max(m))

    Returns:
        best_accuracy : float
            Highest mean CV accuracy
        best_m : int
            Value of m that gave best accuracy
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    best_accuracy = 0
    best_m = None

    for m in m_range:
       # if 2 * m > n_components:
        #    print(f"⚠️ Skipping m = {m} because 2m > n_components")
        #    continue

        fold_accuracies = []

        for train_index, test_index in skf.split(X_alpha, y):
            X_alpha_train, X_alpha_test = X_alpha[train_index], X_alpha[test_index]
            #X_beta_train, X_beta_test = X_beta[train_index], X_beta[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Alpha band CSP
            csp_alpha = CSP(n_components=12)
            X_csp_alpha_train_full = csp_alpha.fit_transform(X_alpha_train, y_train)
           # X_csp_alpha_train = np.concatenate([
           #     X_csp_alpha_train_full[:, :m],
            #    X_csp_alpha_train_full[:, -m:]
           # ], axis=1)

            X_csp_alpha_test_full = csp_alpha.transform(X_alpha_test)
           # X_csp_alpha_test = np.concatenate([
           #     X_csp_alpha_test_full[:, :m],
            #    X_csp_alpha_test_full[:, -m:]
            #], axis=1)

            # Beta band CSP
            #csp_beta = CSP(n_components=n_components)
           # X_csp_beta_train_full = csp_beta.fit_transform(X_beta_train, y_train)
            #X_csp_beta_train = np.concatenate([
            #    X_csp_beta_train_full[:, :m],
           #     X_csp_beta_train_full[:, -m:]
           # ], axis=1)

           # X_csp_beta_test_full = csp_beta.transform(X_beta_test)
           # X_csp_beta_test = np.concatenate([
           #     X_csp_beta_test_full[:, :m],
            #    X_csp_beta_test_full[:, -m:]
           # ], axis=1)

            # Log band power features
          #  X_train_feat = np.concatenate([
            #    np.log(np.var(X_csp_alpha_train, axis=1, keepdims=True)),
           #     np.log(np.var(X_csp_beta_train, axis=1, keepdims=True))
           # ], axis=1)

           # X_test_feat = np.concatenate([
           #     np.log(np.var(X_csp_alpha_test, axis=1, keepdims=True)),
           #     np.log(np.var(X_csp_beta_test, axis=1, keepdims=True))
           # ], axis=1)
            X_train_feat = np.log(np.var(X_csp_alpha_train_full, axis=1))
            X_test_feat = np.log(np.var(X_csp_alpha_test_full, axis=1))
            #plot_csp_features(X_train_feat, y_train, title = f"CSP Features (Train Fold, m={m})")
            # LDA classifier
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda.fit(X_train_feat, y_train)
            y_pred = lda.predict(X_test_feat)

            acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(acc)

        mean_acc = np.mean(fold_accuracies)
        print(f"m = {m} → Mean CV Accuracy: {mean_acc:.2f}")

        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_m = m

    print(f"\n🏆 Best m = {best_m} with mean accuracy = {best_accuracy:.2f}")
    return best_accuracy, best_m

def subject_wise_evaluation(subjects, event_dict, m_range=range(6,7), k=6, n_components=10):
    accuracies = []
    used_subjects = []
    trial_counts = []
    best_ms = []

    for subj in subjects:
        print(f"\n--- Subject {subj} ---")
        X_alpha, X_beta, y = load_bci4_subject_data(subj, event_dict)
        if X_alpha is None or len(y) < k:
            print(f"⚠️ Not enough data for subject {subj}. Skipping.")
            continue

        trial_counts.append(len(y))

        acc, best_m = kfold_search_best_m_csp_lbp_cv(
            X_alpha, X_beta, y,
            m_range=m_range,
            k=k,
            n_components=n_components
        )

        accuracies.append(acc)
        best_ms.append(best_m)
        used_subjects.append(subj)
        mean_acc = np.mean(accuracies)

    # Plot accuracies
    plot_subject_accuracies(used_subjects, accuracies, "Subject-wise LDA Accuracy (k-fold CV)")

    # Print summary table
    print("\n📊 Subject-wise Summary:")
    print("mean accuracy:", mean_acc)
    print(f"{'Subject':<8} {'Trials':<8} {'Accuracy':<10} {'Best m'}")
    for s, n, a, m in zip(used_subjects, trial_counts, accuracies, best_ms):
        print(f"{s:<8} {n:<8} {a:.2f}      {m}")

    # Optional: return results
    return used_subjects, accuracies, best_ms

def subject_wise_cross_validation(subjects, event_dict):
    loo = LeaveOneOut()
    subjects = np.array(subjects)
    accuracies = []
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(subjects):
        train_subjects = subjects[train_idx]
        test_subject = subjects[test_idx][0]
        print(f"\n🔄 LOO-CV: Train={train_subjects}, Test={test_subject}")

        X_alpha_train, X_beta_train, y_train = [], [], []
        for subj in train_subjects:
            Xa, Xb, y = load_bci4_subject_data(subj, event_dict)
            if Xa is not None:
                X_alpha_train.append(Xa)
                X_beta_train.append(Xb)
                y_train.append(y)

        X_alpha_train = np.concatenate(X_alpha_train)
        X_beta_train = np.concatenate(X_beta_train)
        y_train = np.concatenate(y_train)

        Xa_test, Xb_test, y_test = load_bci4_subject_data(test_subject, event_dict)

        csp_alpha = CSP(n_components=12)
        csp_beta = CSP(n_components=12)
        csp_alpha.fit(X_alpha_train, y_train)
        csp_beta.fit(X_beta_train, y_train)

        X_csp_train = np.concatenate([
            csp_alpha.transform(X_alpha_train),
            csp_beta.transform(X_beta_train)
        ], axis=1)

        X_csp_test = np.concatenate([
            csp_alpha.transform(Xa_test),
            csp_beta.transform(Xb_test)
        ], axis=1)

        lda = LinearDiscriminantAnalysis()#solver="lsqr", shrinkage="auto")
        lda.fit(X_csp_train, y_train)
        y_pred = lda.predict(X_csp_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Subject {test_subject} accuracy: {acc:.2f}")
        accuracies.append(acc)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    print("\n=== LOO-CV Summary ===")
    print(f"Mean Accuracy: {np.mean(accuracies):.2f}")
    print(f"Std Dev: {np.std(accuracies):.2f}")

    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Left", "Right"], yticklabels=["Left", "Right"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("LOO-CV Confusion Matrix")
    plt.tight_layout()
    plt.show()

     # Accuracy bar plot
    # Plot accuracy per test subject
    plt.figure(figsize=(10, 4))
    plt.bar([str(s) for s in subjects], accuracies, color='cornflowerblue')
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("LOO-CV Accuracy per Test Subject")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def plot_subject_accuracies(subjects, accuracies, title):
    plt.figure(figsize=(10, 5))
    plt.bar([str(s) for s in subjects], accuracies, color='skyblue')
    plt.ylim(0, 1)
    plt.xlabel("Subject")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# --- MAIN ---
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
event_dict = {
    "left_hand": 769,
    "right_hand": 770
}

subject_wise_evaluation(subjects, event_dict)
#subject_wise_cross_validation(subjects, event_dict)
