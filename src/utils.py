# Helper functions like plotting, evaluation, and file operations
# src/utils.py

import matplotlib.pyplot as plt
import numpy as np

# Plot the full ECG signal with detected peaks
# This function takes the signal and the detected peaks as input and plots them
def plot_signal_with_peaks(signal, peaks, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(signal, label='ECG Signal')
    plt.plot(peaks, signal[peaks], 'rx', label='Detected Peaks')  # 'rx' = red X marks
    plt.title('ECG Signal with Detected Peaks')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# Plot the signal with shaded extracted segments
# This function highlights the segments around the detected peaks
# It takes the signal, detected peaks, and a window size to define the segment length.
def plot_segments(signal, peaks, window_size=50, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(signal, label='ECG Signal')

    for p in peaks:
        start = p - window_size
        end = p + window_size
        if start >= 0 and end <= len(signal):
            plt.axvspan(start, end, color='orange', alpha=0.3)  # Shade the segment

    plt.title('ECG Signal with Segmented Regions')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# Plot the features extracted from segments
# This function takes the features and their labels as input and plots them
# It can be useful for visualizing the distribution of features across different segments.
def plot_features(features, labels, save_path=None):
    plt.figure(figsize=(12, 6))
    for i, feature in enumerate(features.T):
        plt.subplot(2, 2, i + 1)
        plt.scatter(range(len(feature)), feature, c=labels, cmap='viridis', label='Feature {}'.format(i + 1))
        plt.title('Feature {}'.format(i + 1))
        plt.xlabel('Segment Index')
        plt.ylabel('Value')
        plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# Save the features and labels to a CSV file
# This function takes the features and labels as input and saves them to a CSV file
# It can be useful for later analysis or machine learning tasks.
def save_features_to_csv(features, labels, file_path):
    import pandas as pd
    df = pd.DataFrame(features, columns=[f'Feature_{i+1}' for i in range(features.shape[1])])
    df['Label'] = labels
    df.to_csv(file_path, index=False)
