# Helper functions like plotting, evaluation, and file operations
# src/utils.py

import matplotlib.pyplot as plt
import numpy as np

# Plot the full ECG signal with detected peaks
# This function takes the signal and the detected peaks as input and plots them
def plot_signal_with_peaks(signal, peaks, save_path=None, wave_labels=None):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(12, 4))
    plt.plot(signal, label='ECG Signal', color='black')
    # Dynamically assign colors for any label
    import itertools
    color_cycle = itertools.cycle(['blue', 'green', 'red', 'purple', 'orange', 'brown', 'magenta', 'cyan', 'olive', 'teal'])
    label_colors = {}
    y_offset = 0.03 * (np.max(signal) - np.min(signal))
    if wave_labels is not None and len(wave_labels) == len(peaks):
        for lab in set(wave_labels):
            if lab and lab not in label_colors:
                label_colors[lab] = next(color_cycle)
        for p, lab in zip(peaks, wave_labels):
            if lab:
                plt.plot(p, signal[p], 'o', color=label_colors[lab], label=f'Label {lab}' if f'Label {lab}' not in plt.gca().get_legend_handles_labels()[1] else "")
                plt.text(p, signal[p] + y_offset, lab, ha='center', va='bottom', color=label_colors[lab], fontsize=9)
            else:
                plt.plot(p, signal[p], 'rx')
    else:
        plt.plot(peaks, signal[peaks], 'rx', label='Detected Peaks')
    plt.title('ECG Signal with Detected Peaks and Wave Labels')
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
    import numpy as np
    import matplotlib.pyplot as plt
    step = max(1, len(features) // 100)  # Show at most 100 points
    indices = list(range(0, len(features), step))
    sampled_features = features[indices]
    sampled_labels = [labels[i] for i in indices]
    plt.figure(figsize=(14, 8))
    for i, feature in enumerate(sampled_features.T):
        plt.subplot(2, 2, i + 1)
        plt.scatter(range(len(feature)), feature, c='blue', label=f'Feature {i + 1}')
        plt.title(f'Feature {i + 1}')
        plt.xlabel('Segment Index')
        plt.ylabel('Value')
        plt.grid(True)
        # label each point, but only if not too many
        if len(feature) <= 30:
            for idx, val in enumerate(feature):
                plt.text(idx, val, sampled_labels[idx], fontsize=6, ha='center', va='bottom')
        else:
            # If too many, skip text labels
            pass
        plt.legend(fontsize=8)
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
    Label codes: {'':0, 'A':1, 'B':2, 'C':3, ...}
    """
    import csv
    # Dynamically assign codes for all unique labels
    unique_labels = sorted(set(lab for lab in label_array if lab))
    label_map = {'': 0}
    for idx, lab in enumerate(unique_labels, 1):
        label_map[lab] = idx
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "ECG_Value", "Label_Code"])
        for i, (val, lab) in enumerate(zip(signal, label_array)):
            writer.writerow([i, val, label_map.get(lab, 0)])

def plot_wave_annotation(signal, label_array, save_path=None, N=None):
    import matplotlib.pyplot as plt
    import numpy as np
    # Only plot non-empty labels
    if N is not None:
        signal = signal[:N]
        label_array = label_array[:N]
    plt.figure(figsize=(14, 8))
    plt.plot(signal, label='ECG Signal', color='black')
    offset_scale = 0.03 * (np.max(signal) - np.min(signal))
    used_positions = set()
    # Assign colors for all unique labels (ignore blanks)
    unique_labels = [lab for lab in label_array if lab]
    color_cycle = plt.cm.get_cmap('tab10', max(1, len(set(unique_labels))))
    label_colors = {lab: color_cycle(i % 10) for i, lab in enumerate(sorted(set(unique_labels)))}
    for i, lab in enumerate(label_array):
        if lab:
            color = label_colors.get(lab, 'gray')
            offset_factor = 1 + (hash(lab) % 3) * 0.7 + (i % 2) * 0.3
            y_val = signal[i] + offset_factor * offset_scale
            while (i, round(y_val, 2)) in used_positions:
                y_val += 0.01 * offset_scale
            used_positions.add((i, round(y_val, 2)))
            plt.plot(i, y_val, 'o', color=color, label=f'Label {lab}' if f'Label {lab}' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.text(i, y_val, lab, ha='center', va='bottom', color=color, fontsize=9)
    plt.title("ECG Wave Annotation")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    # Only show legend for unique labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
