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
# Assign a letter A,B,C… to each unique segment (after rounding to 3 d.p.).
# Reuses the same letter for repeated shapes.
def assign_labels(segments):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_map = {}
    labels = []
    next_idx = 0

    for seg in segments:
        # round to 3 decimals so that tiny float noise doesn’t break identity
        key = tuple(np.round(seg, 3))
        if key not in label_map:
            label_map[key] = letters[next_idx] if next_idx < len(letters) else f"P{next_idx}"
            next_idx += 1
        labels.append(label_map[key])

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
    labels = assign_labels(segments)  # now pattern‐based labels A, B, C, …
    features = extract_features(segments)
    return segments, labels, features


