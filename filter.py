import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, iirnotch
import matplotlib.pyplot as plt

class filter_class:
    def __init__(self, lowcut=4, highcut=20, notch = None, fs=512,  nmb_channels=15, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = butter(order, [low, high], btype='bandpass')
        self.z = np.zeros((nmb_channels, order*2))
        
        if not notch == None: 
            self.Nb, self.Na = iirnotch(notch, 30, float(fs))
            self.Nz = np.zeros((nmb_channels, 2))
            self.filter_notch = True
        else:
            self.filter_notch = False 
        
        self._nmb_channels = nmb_channels
        self.x = np.empty((nmb_channels,1))
        
        for n in range(nmb_channels):
            if self.filter_notch: self.Nz[n, :] = lfilter_zi(self.Nb, self.Na)
            self.z[n, :] = lfilter_zi(self.b, self.a)

    def filter_data(self, eeg):
        self.x = np.empty(eeg.shape)
        
        for n in range(eeg.shape[0]):
            if self.filter_notch: 
                Neeg, self.Nz[n,:] = lfilter(self.Nb, self.Na, eeg[n,:], zi=self.Nz[n,:])
            else: 
                Neeg = eeg[n,:]
            self.x[n,:], self.z[n,:] = lfilter(self.b, self.a, Neeg, zi=self.z[n,:])
        return self.x.copy()
    
    def filter_epoch(self, eeg):
        x = np.empty(eeg.shape)
        for n in range(eeg.shape[0]):
            if self.filter_notch: 
                Notched = lfilter(self.Nb, self.Na, eeg[n,:])
            else: 
                Notched = eeg[n,:]
            x[n,:] = lfilter(self.b, self.a, Notched)
        return x
