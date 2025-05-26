import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Load continuous ECG data (each row = one time sample)
def load_continuous_ecg(path):
    """
    Loads continuous ECG data.
    Handles both multichannel (signal + time) and single-channel (signal only) CSVs.
    
    Parameters
    ----------
    path : str
        Path to CSV file containing multichannel ECG + timestamp.
    
    Returns
    -------
    signal : np.ndarray
        ECG signal, shape (n_samples, n_channels)
    time : np.ndarray
        Time vector, shape (n_samples,)
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] == 1:
        # Only one column: treat as signal only, no time
        signal = df.values.astype(float)
        signal = signal.reshape(-1, 1)  # shape (n_samples, 1)
        time = np.arange(signal.shape[0])
    else:
        signal = df.iloc[:, :-1].values  # All but last column = signal
        time = df.iloc[:, -1].values     # Last column = time
        signal = signal.astype(float)
        time = time.astype(float)
    return signal, time

# Normalize each channel to zero mean and unit variance
def normalize_multichannel(signal):
    return np.array([(ch - np.mean(ch)) / np.std(ch) for ch in signal.T]).T

# Apply moving average smoothing per channel
def smooth_multichannel(signal, window_size=5):
    kernel = np.ones(window_size) / window_size
    return np.array([np.convolve(ch, kernel, mode='same') for ch in signal.T]).T

# Detect peaks on a selected lead
def detect_peaks_on_lead(signal, lead_index=0, height=None, distance=None, prominence=None):
    """
    Apply peak detection to one lead of the ECG signal.
    
    Returns
    -------
    peaks : np.ndarray
        Indices of detected peaks
    properties : dict
        Properties returned by find_peaks
    """
    lead = signal[:, lead_index]
    peaks, properties = find_peaks(lead, height=height, distance=distance, prominence=prominence)
    return peaks, properties
