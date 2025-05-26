# Report: ECG Segmentation and Pattern Discovery

## Introduction

This project addresses the problem of segmenting and labeling ECG signals into meaningful wave components (P, Q, R, S, T, U) and grouping them into higher-level cardiac cycles (heartbeats). The goal is to move beyond fixed-window segmentation by using an evolutionary algorithm to discover repetitive structures and improve interpretability for downstream analysis.

## Methodology

### Signal Preprocessing

- **Loading:** ECG data is loaded from CSV files, with each row representing a time sample and columns for channels and timestamps.
- **Normalization:** Each channel is normalized to zero mean and unit variance.
- **Smoothing:** A moving average filter is applied to reduce noise.

### Peak Detection

- Peaks are detected on a selected ECG lead using amplitude thresholding and minimum distance constraints.

### Wave Classification Logic

- Detected peaks are heuristically labeled as R-peaks.
- Neighboring peaks are assigned as Q, S, P, and T waves based on their position relative to R-peaks.
- This results in a label array marking the position and type of each detected wave.

### Grouping into Heartbeats

- Labeled waves are grouped into heartbeats (cardiac cycles) by associating each R-peak and its neighboring P, Q, S, T waves.
- Output is saved as a JSON file for further analysis.

### Evolutionary Segmentation Logic

- **Genome Representation:** Each individual is a tuple of segment boundaries (start, end indices).
- **Fitness Function:** Combines peak amplitude, standard deviation, entropy, and correlation with a synthetic R-wave template to favor segments that resemble real cardiac cycles.
- **Selection:** Top-performing individuals are selected based on fitness.
- **Crossover:** Segment boundaries are swapped between parents.
- **Mutation:** Segment boundaries are randomly perturbed.
- **Advantage:** This approach adapts to the data, finding segments that best match repetitive cardiac patterns, outperforming fixed-window methods.

### Feature Extraction

- For each labeled wave, features such as mean, standard deviation, max, and min are extracted from a window around the wave.
- Features are normalized and visualized.

### Clustering (Optional)

- Features from evolutionary segments can be discretized and clustered to discover recurring patterns.

## How to Run the Code

- Place your ECG CSV file in the `data/` directory.
- Run the main script:
  ```
  python src/main.py
  ```
- Output files (plots, CSVs, JSON) will be saved in the `results/` directory.
- Parameters (e.g., smoothing window, peak threshold) can be adjusted in `main.py` or passed as arguments if extended.

## Results & Discussion

- The evolutionary algorithm finds segments that align with cardiac cycles, as visualized in the output plots.
- Labeled waveforms and grouped heartbeats are saved for inspection.
- The approach is robust to signal variability and does not require fixed window sizes.

## Assumptions & Limitations

- Wave labeling is heuristic and may not capture all physiological nuances.
- Grouping is based on detected peaks; missed or extra peaks may affect grouping.
- The evolutionary algorithm is stochastic and may yield different results on each run.

## Dataset Explanation

- The dataset should be a CSV with ECG channels and a timestamp column.
- Example files are provided in the `data/` directory.

---
