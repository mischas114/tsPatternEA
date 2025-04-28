# Performs segmentation of the time series based on heuristic or detected events
import numpy as np

# Handles segmentation of the time series based on heuristic or detected events
# Cuts windows around detected peaks (your main segmentation).
def extract_segments(signal, peaks, window_size=50):
    segments = []
    for p in peaks:
        start = p - window_size
        end = p + window_size
        # Ensure we don't go out of bounds
        if start >= 0 and end <= len(signal):
            segment = signal[start:end]
            segments.append(segment)
    return np.array(segments)

# Handles assignment of labels to segments
# Assigns a naive label heartbeat to each segment (for now) -> Should be replaced with a more sophisticated method later.
def assign_labels(segments, label="heartbeat"):
    labels = [label] * len(segments)
    return labels

# Handles extraction of features from segments
# Extracts simple features (mean, std, max, min) from each segment (for later clustering or EA).
def extract_features(segments):
    features = []
    for segment in segments:
        mean = np.mean(segment)
        std = np.std(segment)
        max_val = np.max(segment)
        min_val = np.min(segment)
        features.append([mean, std, max_val, min_val])
    return np.array(features)

# Handles segmentation of the time series based on heuristic or detected events
# Pipeline function that combines the three steps together (extract → label → feature).
def segment_ecg(signal, peaks, window_size=50):
    segments = extract_segments(signal, peaks, window_size)
    labels = assign_labels(segments)
    features = extract_features(segments)
    return segments, labels, features


