# Transforms time series segments into symbolic/discrete representations
import numpy as np

# Discretize continuous features into bins
def discretize_features(features, n_bins=5):
    discretized = np.zeros_like(features)
    for i in range(features.shape[1]):
        feature = features[:, i]
        bins = np.linspace(np.min(feature), np.max(feature), n_bins + 1)
        discretized[:, i] = np.digitize(feature, bins) - 1  # bin indices start at 0
    return discretized