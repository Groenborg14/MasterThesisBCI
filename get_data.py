import mne
import numpy as np
from filter import filter_class
import os


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
        #"Elbow extend":   2,
        #"Supination"  :   3,
        #"Pronatinon"  :   4,
        #"Hand close"  :   5,
        #"Hand open"   :   6,
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


#def get_data(subjects, run):
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
    y = epochs_alpha.events[:, 2]  # convert event_id 1-7 to labels 0-6
    #y = np.where(y == 1, 0, 1)
    mask = np.isin(y, [1, 3, 6,7])  # Keep only events 1, 3, 6 and 7
    X_alpha = X_alpha[mask]
    X_beta = X_beta[mask]
    y = y[mask]  # Convert to 0-3 labels
    event_map = {1: 0, #2:1, 3: 2, 4:3, 5:4, 6: 5, 
                 7: 1}
    y = np.vectorize(event_map.get)(y)  # Map events to new labels
    
    #print("y =", y," shape = ", y.shape)

    
    return X_alpha,X_beta, y


def get_data(subjects, runs):
    X_alpha_all = []
    X_beta_all = []
    y_all = []

    for subj in subjects:
        for run in runs:
            filepath = f"c:/Users/Christian/Desktop/Master Thesis/MI_data/motorimagination_subject{subj}_run{run}.gdf"
            if not os.path.exists(filepath):
                print(f"⚠️ Missing file: {filepath}")
                continue

            raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False,include=(
                "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                "P3", "P1", "Pz", "P2", "P4", "F1", "Fz", "F2"
            ))
            raw.set_eeg_reference('average')
            raw.apply_function(lambda x: x * 1e-6)
            raw.pick_types(eeg=True)
            raw.interpolate_bads()

            # Replace NaN/inf in raw
            raw._data[np.isnan(raw._data)] = 0
            raw._data[np.isinf(raw._data)] = 0

            events, _ = mne.events_from_annotations(raw)
            tmin, tmax = 0.5, 2.5

            filtered_alpha = filter_class(8, 20).filter_data(raw._data)
            filtered_beta  = filter_class(12, 30).filter_data(raw._data)

            raw_alpha = mne.io.RawArray(filtered_alpha, raw.info)
            raw_beta = mne.io.RawArray(filtered_beta, raw.info)

            event_dict = {
            #"Elbow_flex"  :   1,
            #"Elbow extend":   2,
            #"Supination"  :   3,
            #"Pronatinon"  :   4,
            #"Hand close"  :   5,
            "Hand open"   :   6,
            "Rest"        :   7
            }

            epochs_alpha = mne.Epochs(raw_alpha, events, event_id=event_dict, tmin=tmin, tmax=tmax,
                                      proj=True, baseline=None, preload=True)
            epochs_beta = mne.Epochs(raw_beta, events, event_id=event_dict, tmin=tmin, tmax=tmax,
                                     proj=True, baseline=None, preload=True)

            y = epochs_alpha.events[:, 2]
            mask = np.isin(y, [1, 3, 6, 7])
            X_a = epochs_alpha.get_data()[mask]
            X_b = epochs_beta.get_data()[mask]
            y = y[mask]
            y = np.vectorize({6: 0, 7: 1}.get)(y)

            X_alpha_all.append(X_a)
            X_beta_all.append(X_b)
            y_all.append(y)

    X_alpha = np.concatenate(X_alpha_all, axis=0)
    X_beta = np.concatenate(X_beta_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    return X_alpha, X_beta, y



