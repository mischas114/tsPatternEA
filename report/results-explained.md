# Results Explained: Output Files of the ECG-Processing Pipeline

The pipeline produces several key artefacts. Together they document **what the algorithm did**, **why it made each decision**, and **how you can reproduce or extend the analysis**.

---

## 1 · `all_features.csv`

A tidy, machine-readable table where **each row is one wave** (P, Q, R, S, T, U, ...), and the columns are the four statistics extracted:

| Column      | Meaning                    | Example use-case         |
| ----------- | -------------------------- | ------------------------ |
| `Feature_1` | Segment mean               | Baseline drift detection |
| `Feature_2` | Segment standard deviation | Rhythm variability       |
| `Feature_3` | Maximum amplitude          | Arrhythmia screening     |
| `Feature_4` | Minimum amplitude          | ST-segment depression    |
| `Label`     | Wave class (P, Q, R, ...)  | Supervised ML target     |

The label column may contain canonical wave names (e.g., R1, S1, T1, A2) or numeric/other codes for unclassified or special cases. Because the file is **row-per-wave**, you can feed it directly into scikit-learn or a statistical package without reshaping.  
Parameters that influenced these numbers (e.g., `smoothing_window`, `peak_threshold_factor`, `peak_distance`) are recorded in the metadata for perfect reproducibility.

---

## 2 · `all_features.png` — _Feature dispersion plots_ (_Fig. 1_)

Each of the four sub-plots in **Fig. 1** shows the dispersion of one feature across successive heart-beat segments (x-axis = segment index). The label for each segment is shown above the point if the number of segments is small enough for clarity.

_What to look for_

- **Tight, nearly horizontal clouds** indicate stable morphology.
- **Outliers** flag atypical beats that may merit manual review or could bias a classifier if not handled.
- **Labels** on points help identify which wave or segment is responsible for an outlier.

---

## 3 · `all_peaks.png` — _Full-length signal with detected peaks_ (_Fig. 2_)

**Fig. 2** shows the **normalized** ECG (mean 0, unit variance). Every detected summit is over-plotted:

- **Blue R1, R2** – ventricular peaks
- **Cyan S1, S2** – S-waves
- **Green A2** – auxiliary or unclassified waves
- **Magenta, orange, etc.** – other detected waves (e.g., T1, numeric or special labels)

The legend shows the mapping of label to color. This figure is your quickest “smoke-test”: if the colours land on the wrong humps, tweak the `peak_distance` or threshold. The pipeline now detects both upward and downward peaks for more comprehensive wave identification.

---

## 4 · `evolutionary_segment.png` — _Adaptive vs. fixed segmentation_ (_Fig. 3_)

In **Fig. 3** the **red ribbon** shows the _segment that the evolutionary algorithm considered most informative_ for downstream tasks (e.g., classification).  
The **blue sliver** overlays a traditional fixed window for comparison.

_Why it matters_

- **Adaptive selection** compacts training data yet retains the full P-Q-R morphology.
- You can serialise just this slice to deploy a lightweight model on edge devices.
- The segment is chosen using a fitness function that combines statistical and shape-based metrics.
- The segment indices are recorded in `metadata.json` for reproducibility.

---

## 5 · `wave_annotation.png` and `wave_zoomed.png` — _Label verification views_ (_Figs. 4 & 5_)

- **Fig. 4** – the entire trace with coloured dots and dashed verticals at every wave index (global sanity check).
- **Fig. 5** – a zoom on the first 2 000 samples for _pixel-level_ inspection of label alignment.

The combination lets you confirm both _macro_ correctness (no missing beats) and _micro_ precision (markers on true local maxima). The annotation now uses per-sample wave labels for more detailed inspection.

---

## 6 · `heartbeats.json`

A lightweight list where **each dictionary is one cardiac cycle**:

```json
[
  {"R": 116, "S": 170, "T": 364},
  {"A": 756, "R": 914, "S": 969},
  {},
  ...
]
```

Each entry contains the sample indices for each detected wave in a heartbeat, making it easy to analyze or visualize cardiac cycles. Empty dictionaries indicate cycles where no canonical waves were detected.

---

## 7 · `wave_annotation.csv`

A CSV file mapping each sample to its wave label code (e.g., 0 for no label, 4 for R1, 6 for S1, etc.), allowing for easy downstream analysis or visualization. The mapping of codes to labels is consistent with the annotation and plotting.

---

## 8 · `metadata.json`

A JSON file recording all key parameters and settings used in the run (e.g., smoothing window, peak detection thresholds, evolutionary algorithm settings, segment indices, etc.), ensuring full reproducibility. This file also records the number of detected peaks, segment indices, and clustering settings.

---

## 9 · `raw_data.npz`

A compressed file containing the raw ECG signal and any intermediate arrays, useful for re-running analyses or sharing data without reprocessing from scratch.

---

These artefacts together provide a transparent, reproducible, and extensible record of the ECG analysis pipeline, supporting both clinical and research applications.
