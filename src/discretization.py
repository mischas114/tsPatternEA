# Transforms time series segments into symbolic/discrete representations
import numpy as np

# Discretize continuous features into bins
def discretize_features(features, n_bins=5):
    """
    Discretize continuous features into symbolic bins.
    
    Use this method when you want to transform continuous data into a fixed number of discrete bins
    for symbolic representation or further processing.
    
    Parameters:
    - features: np.ndarray, continuous feature matrix
    - n_bins: int, number of bins to divide the data into
    
    Returns:
    - np.ndarray, discretized feature matrix
    """
    discretized = np.zeros_like(features)
    for i in range(features.shape[1]):
        feature = features[:, i] # Select the i-th feature
        bins = np.linspace(np.min(feature), np.max(feature), n_bins + 1)
        discretized[:, i] = np.digitize(feature, bins) - 1  # bin indices start at 0
    return discretized

def label_by_clustering(features, method='kmeans', n_clusters=3):
    """
    Assign cluster labels to feature vectors using a clustering algorithm.
    
    Use this method when you want to group similar patterns in the data into clusters
    for pattern recognition or grouping purposes.
    
    Parameters:
    - features: np.ndarray, feature matrix
    - method: str, clustering method ('kmeans' supported by default)
    - n_clusters: int, number of clusters to form
    
    Returns:
    - np.ndarray, cluster labels for each feature vector
    """
    from sklearn.cluster import KMeans
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, n_init=10)
        labels = model.fit_predict(features)
        return labels
    # Extend with other clustering methods if desired
    return None