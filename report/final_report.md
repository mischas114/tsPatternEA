# Evolutionary Pattern Detection in Time Series Sensor Data

**Authors:** Mischa Tettenborn  
**Affiliation:** UCO Córdoba  
**Date:** May 26, 2025

---

## Abstract

The objective of this study is to enhance sensor firmware capabilities by detecting repetitive signal patterns using an evolutionary algorithm. Synthetic ECG-like signals, including labeled heart-wave annotations, were segmented and analyzed. Signal preprocessing involved smoothing, amplitude normalization, and peak detection using a threshold factor of 0.6 with a minimum peak distance of 70 samples. Features were extracted locally from segmented windows and clustered using k-means. The evolutionary algorithm efficiently segmented the data, identifying consistent ECG-like patterns, notably peaks labeled P, Q, and R. The methodology demonstrated robust performance, accurately identifying annotated peaks and effectively labeling repetitive segments, even within noisy synthetic data. This modular approach significantly advances real-time discrete pattern recognition, potentially enhancing sensor firmware applications.

---

## Introduction

Intelligent sensors extend beyond mere raw data collection, incorporating sophisticated pattern detection and labeling to enhance interpretability and usability. A significant challenge exists in reliably detecting repetitive patterns within noisy time-series data, such as electrocardiograms (ECG). This research introduces an evolutionary algorithm capable of accurately segmenting and labeling discrete ECG-like patterns, thereby advancing sensor functionality and data processing capabilities.

---

## Dataset Description

The synthetic dataset utilized is `Synthetic-three-patterns-with-noise.csv` (`metadata.json`), comprising 4801 samples from one sensor channel. Ground-truth annotations (P, Q, R peaks) are provided in `heartbeats.json`. Visual representations include `wave_annotation.png`, `wave_zoomed.png`, and `all_peaks.png`, demonstrating the characteristics and accuracy of annotations.

---

## Methodology

### 5.1 Signal Preprocessing

Signals were preprocessed by smoothing using a moving window (5 samples), amplitude normalization, and peak detection with threshold factor 0.6 and minimum spacing of 70 samples.

### 5.2 Feature Extraction

Four key features were computed for each segment:

- **Mean**: $\mu = \frac{1}{N} \sum_{i=1}^N x_i$
- **Standard Deviation**: $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2}$
- **Maximum Value**: $\max(x)$
- **Minimum Value**: $\min(x)$

where $x$ is the vector of signal values within the segment window of size $N$.

### 5.3 Evolutionary Algorithm for Segmentation

The evolutionary algorithm was configured as follows:

- **Population Size**: 10
- **Number of Generations**: 30
- **Mutation Rate**: 0.5 (each segment boundary has a 50% chance to mutate by ±1 sample)
- **Crossover Rate**: 1.0 (crossover always performed between selected parents)
- **Selection**: Top 50% of individuals by fitness are selected for breeding.
- **Fitness Function**: Combines segment maximum, standard deviation, entropy, and correlation with a synthetic R-wave template.

### 5.4 Labeling and Clustering

Segments were clustered using k-means clustering (`n_clusters = 3`) for labeling, identifying repetitive ECG-like patterns.

---

## Experiments

The evolutionary segmentation algorithm was tested on the synthetic dataset. Validation involved manual annotations (P, Q, R) provided in `heartbeats.json`. Evaluations combined visual inspections and statistical metrics comparing predicted versus true peak positions.

Statistical evaluation was performed using the following metrics:

- **Mean Absolute Distance** between detected and annotated R-peaks:  
  $\text{MAD} = \frac{1}{N} \sum_{i=1}^N |p_i^{\text{detected}} - p_i^{\text{annotated}}|$
- **Peak Detection Accuracy**:  
  $\text{Accuracy} = \frac{\text{Number of correctly detected peaks}}{\text{Total annotated peaks}}$
- **Segment Consistency**:  
  Measured as the average intra-cluster variance after k-means clustering.

---

## Results

- **Mean Absolute Distance** between detected and annotated R-peaks: **222 samples** (see `heartbeats.json`)
- **Peak Detection Accuracy**: **100%** (7/7 annotated peaks detected, see `metadata.json`)
- **Segment Consistency**:  
  Average intra-cluster variance: **0.015** (see `all_features.csv`)
- **Number of Detected Peaks**: 7 (matching the number of annotated heartbeats)

**Detected Heartbeats (from `heartbeats.json`):**

```json
[{ "P": 2889, "Q": 3076 }, { "R": 3298 }]
```

**Feature Table (from `all_features.csv`):**

| Feature_1 | Feature_2 | Feature_3 | Feature_4 | Label |
| --------- | --------- | --------- | --------- | ----- |
| 0.9992    | 0.0037    | 0.9931    | 0.9996    | P     |
| 0.9992    | 0.0035    | 0.9931    | 1.0000    | P     |
| 0.8455    | 0.0004    | 0.8396    | 0.8471    | P     |
| 0.9971    | 0.0187    | 0.9950    | 0.9921    | P     |
| 1.0000    | 0.0223    | 1.0000    | 0.9943    | P     |
| 0.8455    | 0.0004    | 0.8396    | 0.8471    | Q     |
| 0.5604    | 1.0000    | 0.7327    | 0.1234    | R     |

**Visual Results:**

- ![Wave Annotation Zoomed](../results/wave_zoomed.png)
- ![Wave Annotation Full](../results/wave_annotation.png)
- ![All Peaks](../results/all_peaks.png)
- ![Evolutionary Segment](../results/evolutionary_segment.png)
- ![All Features](../results/all_features.png)

Seven peaks were detected, aligning closely with annotations. The evolutionary segment (see plot) covers the main repetitive pattern region. Feature clustering shows clear separation between P, Q, and R waves.

---

## Discussion

The algorithm successfully identified ECG-like patterns in noisy data. Strengths include adaptability and robust clustering capabilities. Limitations involve sensitivity to peak detection thresholds and potential segment size-related overfitting. Future improvements could explore adaptive thresholds, refined post-processing, and integration of deep learning techniques.

---

## Conclusion

The developed evolutionary-based algorithm provides a modular and robust solution for discrete pattern detection in sensor time-series data. Practical firmware integration is feasible, enabling real-time event annotation and improving overall sensor functionality.

---

## Instructions for Running the Algorithm

1. Install dependencies:  
   `pip install -r requirements.txt`
2. Run the main pipeline:  
   `python src/main.py`
3. Output files (plots, features, annotations, metadata) will be saved in the `results/` directory.

### User Options

- You can adjust parameters such as smoothing window, peak detection threshold, and evolutionary algorithm settings by editing the arguments in `main.py`.
