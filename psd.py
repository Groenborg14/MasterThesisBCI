import numpy as np
import scipy.signal

class PSDFilter:
    def __init__(self, fs=250, bands=None):
        """
        Initializes the PSD filter.

        Parameters:
        - fs: Sampling frequency (default: 250 Hz).
        - bands: List of frequency bands (default: common EEG bands).
        """
        self.fs = fs  # Sampling frequency (Hz)
        self.bands = bands if bands else {
            "alpha": (8, 12),
            "beta": (12, 30),
            
        }

    def compute_psd(self, x):
        """
        Computes the PSD of the CSP-filtered EEG signal using Welch’s method.
        
        Parameters:
        - x: CSP-transformed signal (shape: trials x channels x time).

        Returns:
        - psd_features: Extracted PSD features for classification.
        """
        n_trials, n_channels, n_samples = x.shape
        psd_features = []

        for trial in range(n_trials):
            trial_features = []

            for channel in range(n_channels):
                # Compute Power Spectral Density (PSD) using Welch’s method
                f, Pxx = scipy.signal.welch(x[trial, channel], fs=self.fs, nperseg=n_samples//2)

                # Extract power in specific bands
                band_powers = []
                for band, (fmin, fmax) in self.bands.items():
                    band_power = np.mean(Pxx[(f >= fmin) & (f <= fmax)])
                    band_powers.append(band_power)

                trial_features.extend(band_powers)

            psd_features.append(trial_features)

        return np.array(psd_features)



def GetRawData(subjects, run):
    
    fpath_subject = []
    fpath_run     = []
    fpath         = []
    k             = 0

    for i in subjects:
        print(i)
        fpath_subject_temp = r"c:\Users\Christian\Desktop\Master Thesis\MI_data\motorimagination_subject"+str(i)
        fpath_subject.append(fpath_subject_temp)
        print(fpath_subject,len(fpath_subject))

    for i in run:
        fpath_run_temp = r"_run"+str(i)+r".gdf"
        fpath_run.append(fpath_run_temp)
        print(fpath_run,len(fpath_run))

    for i in fpath_subject:
        print("im in a for loop")
        for k in fpath_run:
            print("im in while loop")
            fpath_temp = i+k
            fpath.append(fpath_temp)
            print(fpath)
    

    raw = mne.concatenate_raws([mne.io.read_raw_gdf(f, 
                        eog=['eog-l', 'eog-m', 'eog-r'],
                        preload=True, 
                        include= ("C5","C3","C1","Cz","C2", "C4", "C6" ,"P3","P1","Pz","P2","P4")
                        #exclude=('thumb_near', 'thumb_far', 'thumb_index', 'index_near', 'index_far', 'index_middle', 'middle_near', 'middle_far', 'middle_ring', 'ring_near', 'ring_far', 'ring_little', 'litte_near', 'litte_far', 'thumb_palm', 'wrist_bend', 'roll', 'pitch', 'gesture', 'handPosX', 'handPosY', 'handPosZ', 'elbowPosX', 'elbowPosY', 'elbowPosZ', 'ShoulderAdductio', 'ShoulderFlexionE', 'ShoulderRotation', 'Elbow', 'ProSupination', 'Wrist', 'GripPressure','armeodummy')
                        ) for f in fpath])
    
    return raw