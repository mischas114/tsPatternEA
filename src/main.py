# src/main.py
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
from segmentation import segment_ecg
from discretization import discretize_features
from evolutionary import mutate, crossover
from utils import (
    plot_signal_with_peaks,
    plot_segments,
    plot_features,
    save_features_to_csv,
)

##############################################################################
# Main                                                                       #
##############################################################################

def main() -> None:
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

    # ---------- Segmentation ---------------------------------------------
    segments, labels, features = segment_ecg(signal, peaks, window_size=70)
    if features.size == 0 or features.ndim < 2:
        raise RuntimeError("Segmentation produced no valid windows – adjust window_size?")

    plot_segments(
        signal,
        peaks,
        window_size=70,
        save_path=os.path.join(output_dir, "all_segments.png"),
    )

    # use numeric colour IDs (all‑zero because every segment = "heartbeat")
    colour_ids = np.zeros(len(labels), dtype=int)
    plot_features(
        features,
        colour_ids,
        save_path=os.path.join(output_dir, "all_features.png"),
    )

    save_features_to_csv(
        features,
        labels,
        file_path=os.path.join(output_dir, "all_features.csv"),
    )

    # ---------- Discretisation + EA toy ops ------------------------------
    disc = discretize_features(features, n_bins=5)
    save_features_to_csv(disc, labels, os.path.join(output_dir, "all_disc.csv"))

    mut = mutate(disc, mutation_rate=0.1)
    child1_list, child2_list = [], []
    for row in range(disc.shape[0]):
        c1, c2 = crossover(disc[row : row + 1], mut[row : row + 1])
        child1_list.append(c1.squeeze())
        child2_list.append(c2.squeeze())

    save_features_to_csv(mut, labels, os.path.join(output_dir, "all_mut.csv"))
    save_features_to_csv(np.array(child1_list), labels, os.path.join(output_dir, "all_child1.csv"))
    save_features_to_csv(np.array(child2_list), labels, os.path.join(output_dir, "all_child2.csv"))


if __name__ == "__main__":
    main()
