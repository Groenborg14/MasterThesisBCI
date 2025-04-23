import mne
import numpy as np
from filter import filter_class



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
            return None  # or handle the error as needed

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


def create_epochs(raw, events, event_id, tmin, tmax):

    # Define the event dictionary
    # The keys are the event names and the values are the corresponding event IDs
    event_dict = {
        "Elbow_flex"  :   1,
        "Elbow extend":   2,
        "Supination"  :   3,
        "Pronatinon"  :   4,
        "Hand close"  :   5,
        "Hand open"   :   6,
        "Rest"        :   7
    }
    # Filter data in alpha and beta band
    filtered_data_alpha = filter_class(lowcut=8,highcut=12).filter_data(raw._data)
    filtered_data_beta = filter_class(lowcut=12,highcut=30).filter_data(raw._data)
    print(filtered_data_alpha.shape)

    # Turn filtered data back into a mne raw object instead of numpy array
    filtered_raw_alpha = mne.io.RawArray(filtered_data_alpha, raw.info)
    filtered_raw_beta = mne.io.RawArray(filtered_data_beta, raw.info)

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
    
    return epochs_alpha, epochs_beta


def get_data(subjects, run):
    raw = GetRawData(subjects, run)

    # Correcting the data to uV and setting the reference to average 
    raw.apply_function(lambda x: x * 1e-6)
    raw.set_eeg_reference(ref_channels='average')

    # Replace NaN or inf values in the raw data
    raw_data = raw._data  # Access the raw EEG data as a NumPy array
    raw_data[np.isnan(raw_data)] = 0  # Replace NaN values with 0
    raw_data[np.isinf(raw_data)] = 0  # Replace inf values with 0

    # Getting the events from the annotations of the raw EEG data
    events,event_id = mne.events_from_annotations(raw)
    tmin = 1
    tmax = 3
    epochs_alpha, epochs_beta = create_epochs(raw, events, event_id, tmin, tmax)
    
    # Extract data and labels from epochs
    X_alpha = epochs_alpha.get_data()  # shape: (n_trials, n_channels, n_times)
    X_beta = epochs_beta.get_data()  # shape: (n_trials, n_channels, n_times)
    y = epochs_alpha.events[:, 2] - 1  # convert event_id 1-7 to labels 0-6

    
    return X_alpha,X_beta, y






