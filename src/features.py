import numpy as np

def extract_wave_features(signal, label_array, half_window=3):
    """
    Extract features for each labeled wave. Returns (features, wave_labels)
    where features[i] correspond to wave wave_labels[i].
    """
    features = []
    wave_labels = []
    for i, lab in enumerate(label_array):
        if lab:
            start = max(i - half_window, 0)
            end = min(i + half_window + 1, len(signal))
            window = signal[start:end]
            if window.size > 0:  # ensure valid window
                mean = np.mean(window)
                std = np.std(window)
                max_val = np.max(window)
                min_val = np.min(window)
                features.append([mean, std, max_val, min_val])
                wave_labels.append(lab)
    return np.array(features), wave_labels

def extract_features(segments):
    features = []
    for segment in segments:
        mean = np.mean(segment)
        std = np.std(segment)
        max_val = np.max(segment)
        min_val = np.min(segment)
        features.append([mean, std, max_val, min_val])
    return np.array(features)