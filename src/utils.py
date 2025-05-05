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
def plot_segments(signal, peaks, labels, window_size=50, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(signal, label='ECG Signal')
    y_offset = 0.10 * (np.max(signal) - np.min(signal))

    for p, label in zip(peaks, labels):
        # remove colored shading
        # ensure all peaks get labeled
        if 0 <= p < len(signal):
            plt.text(
                p, signal[p] + y_offset, label,
                ha='center', va='bottom', color='black'
            )

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
        plt.scatter(range(len(feature)), feature, c='blue', label=f'Feature {i + 1}')
        plt.title('Feature {}'.format(i + 1))
        plt.xlabel('Segment Index')
        plt.ylabel('Value')
        plt.grid(True)
        # label each point
        for idx, val in enumerate(feature):
            plt.text(idx, val, labels[idx], fontsize=6, ha='center', va='bottom')

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

def save_annotation_to_csv(signal, label_array, file_path):
    """
    Save annotated signal to CSV: [index, value, label_code].
    Label codes: {'':0, 'P':1, 'Q':2, 'R':3, 'S':4, 'T':5, 'U':6}
    """
    label_map = {'':0, 'P':1, 'Q':2, 'R':3, 'S':4, 'T':5, 'U':6}
    import csv
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "ECG_Value", "Label_Code"])
        for i, (val, lab) in enumerate(zip(signal, label_array)):
            writer.writerow([i, val, label_map.get(lab, 0)])

def plot_wave_annotation(signal, label_array, save_path=None):
    """
    Overlay wave labels on the ECG signal with distinct colors/markers,
    vertical lines for reference, and slight y-offsets to prevent overlap.
    """
    label_colors = {'P':'blue', 'Q':'green', 'R':'red', 'S':'purple', 'T':'orange', 'U':'brown'}
    plt.figure(figsize=(12, 4))
    plt.plot(signal, label='ECG Signal', color='black')
    offset_scale = 0.02 * (np.max(signal) - np.min(signal))

    for i, lab in enumerate(label_array):
        if lab:
            color = label_colors.get(lab, 'gray')
            y_val = signal[i] + (offset_scale if lab != 'R' else 2*offset_scale)
            plt.scatter(i, y_val, color=color, marker='o', s=40, zorder=3)
            plt.axvline(x=i, color=color, linewidth=0.5, linestyle='--', alpha=0.5)
            plt.text(
                i, y_val + offset_scale, lab, ha='center', va='bottom',
                fontsize=8, color=color
            )

    plt.title("ECG Wave Annotation")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
