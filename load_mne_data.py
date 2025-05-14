import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from csp import CSP

print(__doc__)

# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -1.0, 4.0

runs = [6, 10, 14]  # motor imagery: hands vs feet

def load_raw_and_get_epochs(subjects, runs):
    raw_fnames = eegbci.load_data(subjects, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.annotations.rename(dict(T1="hands", T2="feet"))  # as documented on PhysioNet
    raw.set_eeg_reference(projection=True)

    # Apply band-pass filter
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
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





def lda_with_sliding_window(epochs_data, labels, window_size=0.5, step_size=0.1, sfreq=250.0):
    """
    Apply sliding window LDA classification over epochs data.
    
    Parameters:
    - epochs_data: ndarray (n_epochs, n_channels, n_times)
    - labels: ndarray (n_epochs,)
    - window_size: float (seconds)
    - step_size: float (seconds)
    - sfreq: float (sampling frequency in Hz)
    
    Returns:
    - accuracy: float, mean accuracy using majority vote over sliding windows
    """

    n_epochs, n_channels, n_times = epochs_data.shape
    w_size = int(window_size * sfreq)
    w_step = int(step_size * sfreq)
    n_windows = int((n_times - w_size) / w_step) + 1

    csp = CSP(n_components=4)
    csp.fit(epochs_data, labels)

    lda = LinearDiscriminantAnalysis()

    # Split data into training and test (using the same data here for illustration)
    csp_features = []
    for i in range(n_windows):
        start = i * w_step
        stop = start + w_size
        windowed_data = epochs_data[:, :, start:stop]
        csp_transformed = csp.transform(windowed_data)
        csp_features.append(csp_transformed)
    
    csp_features = np.stack(csp_features, axis=1)  # shape: (n_epochs, n_windows, n_features)

    # Flatten features per window for training
    features_for_training = csp_features.mean(axis=1)
    lda.fit(features_for_training, labels)

    # Predict per window and take majority vote
    window_preds = []
    for i in range(n_windows):
        pred = lda.predict(csp_features[:, i, :])
        window_preds.append(pred)

    window_preds = np.stack(window_preds, axis=1)  # shape: (n_epochs, n_windows)
    
    # Majority vote
    from scipy.stats import mode
    final_preds, _ = mode(window_preds, axis=1)
    final_preds = final_preds.ravel()

    accuracy = np.mean(final_preds == labels)
    return accuracy

def lda_with_sliding_window_crossval(X_train, y_train, X_test, y_test, sfreq, window_size=0.5, step_size=0.1):
    csp = CSP(n_components=4)
    csp.fit(X_train, y_train)

    def get_features(X):
        w_size = int(window_size * sfreq)
        w_step = int(step_size * sfreq)
        n_windows = int((X.shape[2] - w_size) / w_step) + 1
        feats = []
        for i in range(n_windows):
            start = i * w_step
            end = start + w_size
            window = X[:, :, start:end]
            feats.append(csp.transform(window))
        feats = np.stack(feats, axis=1)
        return feats

    train_feats = get_features(X_train).mean(axis=1)
    test_feats = get_features(X_test)

    lda = LinearDiscriminantAnalysis()
    lda.fit(train_feats, y_train)

    from scipy.stats import mode
    preds_per_window = np.array([lda.predict(test_feats[:, i, :]) for i in range(test_feats.shape[1])])
    preds_majority = mode(preds_per_window, axis=0)[0].ravel()

    return np.mean(preds_majority == y_test)


def leave_one_subject_out(subjects, runs):
    from sklearn.model_selection import LeaveOneGroupOut

    X_all = []
    y_all = []
    groups = []
    subject_ids = []

    # Load and store data for each subject
    for i, subj in enumerate(subjects):
        epochs, labels = load_raw_and_get_epochs(subj, runs)
        X = epochs.get_data()
        X_all.append(X)
        y_all.append(labels)
        groups.extend([i] * len(labels))
        subject_ids.append(subj)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all)
    groups = np.array(groups)

    logo = LeaveOneGroupOut()
    accs = []
    subject_acc = {}

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_all, y_all, groups)):
        train_subjects = sorted(set(groups[train_idx]))
        test_subject = list(set(groups[test_idx]))[0]
        test_subject_id = subject_ids[test_subject]

        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train subjects: {[subject_ids[i] for i in train_subjects]}")
        print(f"  Test subject: {test_subject_id}")

        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        sfreq = epochs.info['sfreq']  # Can also pull from epochs.info if you pass that along
        acc = lda_with_sliding_window_crossval(X_train, y_train, X_test, y_test, sfreq)

        print(f"  Accuracy for subject {test_subject_id}: {acc:.2f}")
        accs.append(acc)
        subject_acc[test_subject_id] = acc

    mean_acc = np.mean(accs)
    print(f"\nMean LOSO Accuracy: {mean_acc:.2f}")
    return mean_acc, subject_acc

def plot_subject_accuracies(subject_acc):
    subjects = list(subject_acc.keys())
    accuracies = [subject_acc[subj] for subj in subjects]

    plt.figure(figsize=(10, 5))
    plt.bar(subjects, accuracies)
    plt.xlabel("Subject")
    plt.ylabel("Accuracy")
    plt.title("LDA + CSP Classification Accuracy per Subject")
    plt.ylim(0, 1)
    plt.xticks(subjects)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

#epochs_train, labels = load_raw_and_get_epochs(4, runs)
#accuracy = lda_with_sliding_window(epochs_train.get_data(), labels, window_size=0.5, step_size=0.1, sfreq=epochs_train.info['sfreq'])
#print(f"Sliding Window LDA Accuracy: {accuracy:.2f}")

subjects = [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16]
mean_acc, subject_accuracies = leave_one_subject_out(subjects, runs)

print("\nIndividual Subject Accuracies:")
for subj, acc in subject_accuracies.items():
    print(f"  Subject {subj}: {acc:.2f}")

plot_subject_accuracies(subject_accuracies)

