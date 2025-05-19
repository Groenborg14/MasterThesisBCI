import mne
from filter import filter_class
import os
import matplotlib.pyplot as plt
from csp import CSP
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report





def GetRawData(subjects, run):
    import os

    fpath = []

    for subj in subjects:
        subj_path = "c:/Users/Christian/Desktop/Master Thesis/MI_data/motorimagination_subject" + str(subj)
        for r in run:
            full_path = os.path.join(subj_path + f"_run{r}.gdf")
            fpath.append(full_path)
            print(f"Added path: {full_path}")

    # Debug print to verify file existence
    for path in fpath:
        if not os.path.exists(path):
            print(f"⚠️ File does NOT exist: {path}")

    # Load files
    raw = mne.concatenate_raws([
        mne.io.read_raw_gdf(
            f,
            eog=['eog-l', 'eog-m', 'eog-r'],
            preload=True,
            include=(
                "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                "P3", "P1", "Pz", "P2", "P4", "F1", "Fz", "F2"
            )
        )
        for f in fpath
    ])

    return raw

# --- LOAD RAW FILES ---
def load_multi_run(subjects, runs):
    raw_list = []
    for subj in subjects:
        subj_path = f"c:/Users/Christian/Desktop/Master Thesis/MI_data/motorimagination_subject{subj}"
        for run in runs:
            filepath = os.path.join(subj_path + f"_run{run}.gdf")
            if os.path.exists(filepath):
                raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False,include=(
                "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                "P3", "P1", "Pz", "P2", "P4", "F1", "Fz", "F2"
            ))
                raw.set_eeg_reference('average')
                raw.apply_function(lambda x: x * 1e-6)  # Convert to uV
                raw_list.append(raw)
            else:
                print(f"⚠️ Missing file: {filepath}")
    
    mne.channels.equalize_channels(raw_list)

    # Concatenate without introducing BAD boundaries
    raw_combined = mne.concatenate_raws(raw_list, on_mismatch='ignore')

    # Remove 'BAD boundary' annotations (which may cause gaps or NaNs)
    raw_combined.annotations.delete(
        [i for i, desc in enumerate(raw_combined.annotations.description) if 'BAD' in desc]
    )

    return raw_combined

subjects = [13,14]
run = [1,2,3,4,5,6,7,8,9,10]

# Aquiring the data
raw = load_multi_run(subjects,run)

raw.info['bads'] = []


# Correcting the data to uV and setting the reference to average 
raw.apply_function(lambda x: x * 1e-6)
raw.set_eeg_reference(ref_channels = 'average')
raw.pick_types(eeg=True)
raw.interpolate_bads()
# Replace NaN or inf values in the raw data
raw_data = raw._data  # Access the raw EEG data as a NumPy array
#raw_data[np.isnan(raw_data)] = 0  # Replace NaN values with 0
#raw_data[np.isinf(raw_data)] = 0  # Replace inf values with 0

# Getting the events from the annotations of the raw EEG data
events,event_id = mne.events_from_annotations(raw)

# Define the event dictionary
# The keys are the event names and the values are the corresponding event IDs
event_dict = {
    "Elbow_flex"  :   1,
    #"Elbow extend":   2,
    #"Supination"  :   3,
    #"Pronatinon"  :   4,
    #"Hand close"  :   5,
    #"Hand open"   :   6,
    "Rest"        :   7
}

#raw.plot(events=events, start=0, duration=30)

#eeg_data = raw._data()
print(raw.ch_names)
#print(raw.info)
print(raw._data.shape)
#fc = filter_class()

# Filter data in alpha and beta band
filtered_data_alpha = filter_class(lowcut=8,highcut=12).filter_data(raw._data)
filtered_data_beta = filter_class(lowcut=12,highcut=30).filter_data(raw._data)
print(filtered_data_alpha.shape)

# Turn filtered data back into a mne raw object instead of numpy array
filtered_raw_alpha = mne.io.RawArray(filtered_data_alpha, raw.info)
filtered_raw_beta = mne.io.RawArray(filtered_data_beta, raw.info)

#filtered_raw_alpha.plot(events=events, start=0, duration=30)
#filtered_raw_beta.plot(events=events, start=0, duration=30)



tmin = 1
tmax = 3

# Create epochs from the filtered data
epochs_alpha = mne.Epochs(
    filtered_raw_alpha,
    events,
    event_id=event_dict,
    tmin=tmin,
    tmax=tmax,
    proj=True,
    baseline=None,
    preload=True,
)
epochs_beta = mne.Epochs(
    filtered_raw_beta,
    events,
    event_id=event_dict,
    tmin=tmin,
    tmax=tmax,
    proj=True,
    baseline=None,
    preload=True,
)
def print_nan_summary(name, data, labels):
    n_total = data.shape[0]
    has_nan = np.isnan(data).any(axis=(1, 2))
    has_inf = np.isinf(data).any(axis=(1, 2))

    for label in np.unique(labels):
        class_mask = (labels == label)
        total_class = class_mask.sum()
        n_nan = has_nan[class_mask].sum()
        n_inf = has_inf[class_mask].sum()
        print(f"{name} - Class {label}: {total_class} trials | NaNs: {n_nan} | Infs: {n_inf}")

print_nan_summary("Alpha", epochs_alpha.get_data(), epochs_alpha.events[:, 2])
print_nan_summary("Beta", epochs_beta.get_data(), epochs_beta.events[:, 2])

def drop_nan_epochs(epochs):
    data = epochs.get_data()
    mask = ~np.isnan(data).any(axis=(1, 2)) & ~np.isinf(data).any(axis=(1, 2))
    return epochs[mask]

#epochs_alpha = drop_nan_epochs(epochs_alpha)
#epochs_beta = drop_nan_epochs(epochs_beta)

print(epochs_alpha.get_data().shape)
print(epochs_beta.get_data().shape)
#epochs_beta["Elbow_flex"].plot(event_id=event_dict,events=events)

#print(epochs_alpha.get_data().shape)
# Using CSP to filter the data in the two bands


#print(csp_filter_alpha.shape)

# Concatenate the CSP-filtered data from alpha and beta bands

#print(csp_filter_combined.shape)


# Prepare the data



y = epochs_alpha.events[:, 2]  # Labels (class labels from events)
X = np.arange(y.shape[0])  # Features (CSP-transformed data)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the LDA classifier
lda = LinearDiscriminantAnalysis()

alpha_epochs = epochs_alpha.get_data()
beta_epochs = epochs_beta.get_data()

csp_alpha = CSP()
csp_beta = CSP()
csp_filter_alpha = csp_alpha.fit_transform(alpha_epochs[X_train], y_train.copy())
csp_filter_beta = csp_beta.fit_transform(beta_epochs[X_train], y_train.copy())

csp_filter_combined = np.concatenate((csp_filter_alpha, csp_filter_beta), axis=1)  # Concatenate along columns

# Train the LDA classifier
lda.fit(csp_filter_combined, y_train)

# Make predictions on the test set

csp_filter_alpha_test = csp_alpha.transform(alpha_epochs[X_test])
csp_filter_beta_test = csp_beta.transform(beta_epochs[X_test])

csp_filter_combined_test = np.concatenate((csp_filter_alpha_test, csp_filter_beta_test), axis=1)  # Concatenate along columns

y_pred = lda.predict(csp_filter_combined_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, label="True Labels", alpha=0.7)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted Labels", alpha=0.7)
plt.title("LDA Classification Results")
plt.xlabel("Sample Index")
plt.ylabel("Class Label")
plt.legend()
plt.grid(True)
plt.show()