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

