# Entry point to run preprocessing, segmentation, or full EA pipeline

import numpy as np
from preprocessing import load_ecg_data, normalize, smooth, detect_peaks
from segmentation import segment_ecg
from utils import plot_signal_with_peaks, plot_segments, plot_features, save_features_to_csv
import os
from preprocessing import load_ecg_data, normalize, smooth, detect_peaks
from segmentation import segment_ecg
from utils import plot_signal_with_peaks, plot_segments, plot_features, save_features_to_csv
from discretization import discretize_features
from evolutionary import mutate, crossover


def main():
    # Define paths
    data_path = os.path.join ("data", "ecg5000_test.csv")
    output_dir = os.path.join("results")    
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load and preprocess data
    print("Loading ECG data...")
    signal = load_ecg_data(data_path)
    signal = normalize(signal)
    signal = smooth(signal, window_size=5)

    # Step 2: Detect peaks
    print("Detecting peaks...")
    peaks, _ = detect_peaks(signal, height=0.5, distance=50)

    # Step 3: Plot signal with detected peaks
    print("Plotting signal with detected peaks...")
    plot_signal_with_peaks(signal, peaks, save_path=os.path.join(output_dir, "signal_with_peaks.png"))

    # Step 4: Segment the signal
    print("Segmenting signal...")
    segments, labels, features = segment_ecg(signal, peaks, window_size=50)

    # Step 5: Plot segmented regions
    print("Plotting segmented regions...")
    plot_segments(signal, peaks, window_size=50, save_path=os.path.join(output_dir, "segmented_signal.png"))

    # Step 6: Plot extracted features
    print("Plotting extracted features...")
    plot_features(features, labels, save_path=os.path.join(output_dir, "features.png"))

    # Step 7: Save features and labels to CSV
    print("Saving features and labels...")
    save_features_to_csv(features, labels, file_path=os.path.join(output_dir, "features.csv"))

    # Step 8: Discretize features
    print("Discretizing features...")
    discretized_features = discretize_features(features, n_bins=5)
    save_features_to_csv(discretized_features, labels, file_path=os.path.join(output_dir, "discretized_features.csv"))

    # Step 9: Apply evolutionary operations
    print("Applying evolutionary operations...")
    mutated_features = mutate(discretized_features, mutation_rate=0.1)

    # Crossover sample by sample
    child1_list = []
    child2_list = []
    for i in range(discretized_features.shape[0]):
        c1, c2 = crossover(discretized_features[i:i+1, :], mutated_features[i:i+1, :])
        child1_list.append(c1.squeeze())
        child2_list.append(c2.squeeze())

    child1 = np.array(child1_list)
    child2 = np.array(child2_list)

    # Step 10: Save evolved features
    print("Saving evolved features...")
    save_features_to_csv(mutated_features, labels, file_path=os.path.join(output_dir, "mutated_features.csv"))
    save_features_to_csv(child1, labels, file_path=os.path.join(output_dir, "child1_features.csv"))
    save_features_to_csv(child2, labels, file_path=os.path.join(output_dir, "child2_features.csv"))



if __name__ == "__main__":
    main()