# Results Explained: Output Files of the ECG-Processing Pipeline

The pipeline produces nine key artefacts. Together they document **what the algorithm did**, **why it made each decision**, and **how you can reproduce or extend the analysis**.

---

## 1 · `all_features.csv`

A tidy, machine-readable table where **each row is one wave** (P, Q, R …) and the columns are the four statistics we extract:

| Column      | Meaning                    | Example use-case         |
| ----------- | -------------------------- | ------------------------ |
| `Feature_1` | Segment mean               | Baseline drift detection |
| `Feature_2` | Segment standard deviation | Rhythm variability       |
| `Feature_3` | Maximum amplitude          | Arrhythmia screening     |
| `Feature_4` | Minimum amplitude          | ST-segment depression    |
| `Label`     | Wave class (P, Q, R …)     | Supervised ML target     |

Because the file is **row-per-wave**, you can feed it directly into scikit-learn or a statistical package without reshaping.  
Parameters that influenced these numbers (e.g., `smoothing_window = 5`, `peak_threshold_factor = 0.6`) are recorded in the metadata for perfect reproducibility. :contentReference[oaicite:0]{index=0}

---

## 2 · `all_features.png` — _Feature dispersion plots_ (_Fig. 1_)

Each of the four sub-plots in **Fig. 1** (insert image here) shows the dispersion of one feature across successive heart-beat segments (x-axis = segment index).

_What to look for_

- **Tight, nearly horizontal clouds** (segments 0-5) mean the algorithm found stable morphology.
- **Outliers** (segment 6) flag atypical beats that may merit manual review or could bias a classifier if not handled.

---

## 3 · `all_peaks.png` — _Full-length signal with detected peaks_ (_Fig. 2_)

Paste **Fig. 2** immediately below.  
The black trace is the **normalized** ECG (mean 0, unit variance). Every detected summit is over-plotted:

- **Blue P**’s – atrial depolarization
- **Green Q** – ventricular depolarization onset
- **Red R** – ventricular peak

This figure is your quickest “smoke-test”: if the colours land on the wrong humps, tweak the `peak_distance` or threshold. (Seven good peaks were found in this run. :contentReference[oaicite:1]{index=1})

---

## 4 · `evolutionary_segment.png` — _Adaptive vs. fixed segmentation_ (_Fig. 3_)

In **Fig. 3** the **red ribbon** (samples 716-3216) shows the _segment that a genetic algorithm considered most informative_ for downstream tasks (e.g., classification).  
The **blue sliver** overlays a traditional fixed window for comparison.

_Why it matters_

- **Adaptive selection** compacts training data by ~55 % yet retains the full P-Q-R morphology.
- You can serialise just this slice to deploy a lightweight model on edge devices.

---

## 5 · `wave_annotation.png` and `wave_zoomed.png` — _Label verification views_ (_Figs. 4 & 5_)

- **Fig. 4** – the entire trace with coloured dots and dashed verticals at every wave index (global sanity check).
- **Fig. 5** – a zoom on the first 2 000 samples for _pixel-level_ inspection of label alignment.

The combination lets you confirm both _macro_ correctness (no missing beats) and _micro_ precision (markers on true local maxima).

---

## 6 · `heartbeats.json`

A lightweight list where **each dictionary is one cardiac cycle**:

```json
[{ "P": 2889, "Q": 3076 }, { "R": 3298 }]
```
