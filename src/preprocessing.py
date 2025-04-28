# Handles normalization, smoothing, and peak detection
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Load ECG data from CSV file
def load_ecg_data(path, drop_labels=True):
    df = pd.read_csv(path, header=None)
    if drop_labels:
        data = df.iloc[:, 1:].values  # first column is numbers
    else:
        data = df.values
    return data.squeeze()  # convert to 1D array

# Standardize signal to zero mean and unit variance
def normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / std

# Smooth signal with moving average (window_size samples)
def smooth(signal, window_size=5):
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode='same')

# Detect peaks in the signal using scipy's find_peaks
def detect_peaks(signal, height=None, distance=None, prominence=None):
    peaks, props = find_peaks(signal, height=height, distance=distance, prominence=prominence)
    return peaks, props