from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import json
from datetime import datetime


from preprocessing import (
    load_continuous_ecg,
    normalize_multichannel,
    smooth_multichannel,
    detect_peaks_on_lead,
    detect_bipolar_peaks,  # <-- import the new function
)
from segmentation import extract_segments, segment_ecg
from evolutionary import run_evolutionary_segmentation
from labeling import assign_labels, annotate_waves, group_waves_into_heartbeats
from features import extract_wave_features, extract_features
from utils import (
    plot_signal_with_peaks,
    plot_features,
    save_features_to_csv,
    plot_wave_annotation,
    save_annotation_to_csv,
)
from discretization import discretize_features, label_by_clustering


def main(
    data_path: str = "data/ecg-real.csv",
    output_dir: str = "results",
    selected_lead: int = 0,
    smooth_window: int = 5,
    peak_height_factor: float = 0.6,
    peak_distance: int = 70,
    plot_zoom_window: int = 2000,
    enable_evolution: bool = True,
    enable_clustering: bool = True,
) -> None:
    """
    Universal ECG processing pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------- Load & Preprocess ECG ------------------------
    signal, time = load_continuous_ecg(data_path)
    print(f"Loaded {signal.shape[0]} samples across {signal.shape[1]} channels")

    signal = normalize_multichannel(signal)
    signal = smooth_multichannel(signal, window_size=smooth_window)

    # ---------- Peak Detection -------------------------------
    # --- Peak Detection (new, both polarities) --------------------------
    lead_signal = signal[:, selected_lead]
    peaks = detect_bipolar_peaks(lead_signal,
                                 height_frac=0.15,
                                 distance=peak_distance)
    print(f"Detected {len(peaks)} peaks (both polarities).")

    # --- Assign wave labels for peaks for plotting ---
    fs = 250  # Sampling frequency in Hz (adjust if needed)
    label_array = annotate_waves(lead_signal, peaks, fs=fs)
    wave_labels = [label_array[p] if 0 <= p < len(label_array) else '' for p in peaks]

    plot_signal_with_peaks(
        lead_signal, peaks,
        save_path=os.path.join(output_dir, "all_peaks.png"),
        wave_labels=wave_labels
    )

    # Group waves into heartbeats (cardiac cycles)
    heartbeats = group_waves_into_heartbeats(peaks, wave_labels)
    # Save heartbeats to JSON for inspection
    with open(os.path.join(output_dir, "heartbeats.json"), "w") as f:
        json.dump(heartbeats, f, indent=2, default=int)

    plot_wave_annotation(
        lead_signal, label_array,
        save_path=os.path.join(output_dir, "wave_annotation.png"),
    )
    plot_wave_annotation(
        lead_signal, label_array,
        save_path=os.path.join(output_dir, "wave_zoomed.png"),
        N=plot_zoom_window,
    )
    save_annotation_to_csv(
        lead_signal, label_array,
        os.path.join(output_dir, "wave_annotation.csv"),
    )

    # ---------- Feature Extraction ---------------------------
    features, wave_labels = extract_wave_features(lead_signal, label_array)
    if features.size > 0:
        features = features / (np.max(np.abs(features), axis=0, keepdims=True) + 1e-9)
        plot_features(
            features, wave_labels,
            save_path=os.path.join(output_dir, "all_features.png"),
        )
        save_features_to_csv(
            features, wave_labels,
            file_path=os.path.join(output_dir, "all_features.csv"),
        )
    else:
        print("No features extracted (not enough peaks/waves detected). Skipping feature plots.")

    if enable_evolution:
        # ---------- Evolutionary Segmentation -------------------------
        best_segment = run_evolutionary_segmentation(lead_signal)
        best_segment = (int(best_segment[0]), int(best_segment[1]))
        print(f"Best segment: {best_segment}")

        plt.figure(figsize=(12, 4))
        plt.plot(lead_signal, label='ECG Signal')
        plt.axvspan(*best_segment, color='red', alpha=0.3, label='Evolutionary Segment')
        if len(peaks) > 0:
            fixed_start = max(0, peaks[0] - 50)
            fixed_end = min(len(lead_signal), peaks[0] + 50)
            plt.axvspan(fixed_start, fixed_end, color='blue', alpha=0.2, label='Fixed Segment')
        plt.title('ECG Signal with Evolutionary and Fixed Segments')
        plt.xlabel('Sample')
        plt.ylabel('Normalized Amplitude')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "evolutionary_segment.png"))
        plt.close()

        if enable_clustering:
            # ---------- Feature Clustering --------------------------
            segment_data = lead_signal[best_segment[0]:best_segment[1]]
            if len(segment_data) > 10:
                window_size = 10
                stride = 5
                windows = [segment_data[i:i+window_size]
                           for i in range(0, len(segment_data)-window_size+1, stride)]
                window_features = extract_features(windows)
                disc = discretize_features(window_features, n_bins=4)
                cluster_labels = label_by_clustering(window_features, n_clusters=3)
                print(f"Discretized features shape: {disc.shape}")
                print(f"Cluster labels: {cluster_labels}")

    # ---------- Save metadata ---------------------------------------------
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_path": data_path,
        "output_dir": output_dir,
        "selected_lead": selected_lead,
        "smoothing_window": smooth_window,
        "peak_threshold_factor": peak_height_factor,
        "peak_distance": peak_distance,
        "num_samples": int(signal.shape[0]),
        "num_channels": int(signal.shape[1]),
        "num_peaks_detected": int(len(peaks)),
        "wave_annotation_file": os.path.join(output_dir, "wave_annotation.csv"),
        "features_file": os.path.join(output_dir, "all_features.csv"),
        "evolutionary_segmentation": {
            "enabled": enable_evolution,
            "segment": best_segment if enable_evolution else None,
            "segment_file": os.path.join(output_dir, "evolutionary_segment.png") if enable_evolution else None,
        },
        "clustering": {
            "enabled": enable_clustering,
            "n_clusters": 3 if enable_clustering else None,
        }
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {os.path.join(output_dir, 'metadata.json')}")

    # ---------- Save Raw Data for Report -------------------------------
    np.savez(os.path.join(output_dir, "raw_data.npz"),
             signal=signal,
             lead_signal=lead_signal,
             peaks=peaks,
             label_array=label_array,
             features=features if features.size > 0 else [],
             wave_labels=np.array(wave_labels),
             segment=best_segment if enable_evolution else [],
             cluster_labels=cluster_labels if enable_evolution and enable_clustering else [],
             discretized_features=disc if enable_evolution and enable_clustering else []
    )

    print(f"Raw data saved to {os.path.join(output_dir, 'raw_data.npz')}")




if __name__ == "__main__":
    main()
