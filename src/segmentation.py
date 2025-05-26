# Performs segmentation of the time series based on heuristic or detected events
import numpy as np

from features import extract_features
from labeling import assign_labels

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

# Handles segmentation of the time series based on heuristic or detected events
# Pipeline function that combines the three steps together (extract → label → feature).
def segment_ecg(signal, peaks, window_size=50):
    segments = extract_segments(signal, peaks, window_size)
    labels = assign_labels(segments)  # now pattern‐based labels A, B, C, …
    features = extract_features(segments)
    return segments, labels, features


