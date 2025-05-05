"""Entry point for the Phase‑1 **single‑trace** pipeline.

This version treats the whole ECG5000 test set as **one long time‑series**:
    • each 140‑sample row is z‑normalised and smoothed **individually**
      (avoids giant‑spike artefact)
    • the beats are then concatenated → 1‑D array of length N_beats×140

Outputs go to `results/` with global‑trace filenames.
"""
from __future__ import annotations

import os
import numpy as np

from preprocessing import load_ecg_data, normalize, smooth, detect_peaks
from segmentation import annotate_waves, extract_wave_features
from utils import (
    plot_signal_with_peaks,
    plot_features,
    save_features_to_csv,
    plot_wave_annotation,
    save_annotation_to_csv,
)

##############################################################################
# Main                                                                       #
##############################################################################

def main() -> None:
    """
    Main pipeline to load ECG data, preprocess it, detect peaks, segment
    the signal, assign labels, extract features, and save results to disk.
    """
    # ---------- Paths -----------------------------------------------------
    # data_path = os.path.join("data", "ecg5000_test.csv")
    data_path = os.path.join("data", "synthetic_ecg_small.csv")
    output_dir = os.path.join("results")
    os.makedirs(output_dir, exist_ok=True)

    # ---------- Load & per‑beat normalise ---------------------------------
    traces = load_ecg_data(data_path)  # shape (N_beats, 140)
    print(f"Loaded {traces.shape[0]} beats → concatenating into one trace …")

    norm_traces = np.array([smooth(normalize(b), window_size=5) for b in traces])
    signal = norm_traces.flatten()  # length = N_beats * 140

    # ---------- Peak detection on the long trace -------------------------
    thr = 0.6 * np.max(signal)
    peaks, _ = detect_peaks(signal, height=thr, distance=70)  # ≥ half a beat
    print(f"Detected {len(peaks)} peaks in concatenated trace")

    # ---------- Visualise raw trace + peaks ------------------------------
    plot_signal_with_peaks(
        signal,
        peaks,
        save_path=os.path.join(output_dir, "all_peaks.png"),
    )

    # ---------- Wave Annotation ------------------------------------------
    label_array = annotate_waves(signal, peaks)

    # Plot & save annotation
    plot_wave_annotation(
        signal, label_array,
        save_path=os.path.join(output_dir, "wave_annotation.png"),
    )
    save_annotation_to_csv(
        signal, label_array,
        os.path.join(output_dir, "wave_annotation.csv"),
    )

    # ---------- Wave-based Feature Extraction ----------------------------
    features, wave_labels = extract_wave_features(signal, label_array)

    # normalize features for plotting
    features = features / (np.max(np.abs(features), axis=0, keepdims=True) + 1e-9)

    # Plot & save features
    plot_features(
        features,
        wave_labels,
        save_path=os.path.join(output_dir, "all_features.png"),
    )
    save_features_to_csv(
        features,
        wave_labels,
        file_path=os.path.join(output_dir, "all_features.csv"),
    )


if __name__ == "__main__":
    main()
