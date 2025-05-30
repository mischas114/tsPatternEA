# Project Overview: ECG Signal Processing Pipeline

This document explains the main components and workflow of the ECG (Electrocardiogram) signal processing program in simple terms. It is intended for inclusion in the project report.

## What Does the Program Do?

The program processes ECG data (heart signal recordings) to detect important points (waves), segment the signal, extract features, and prepare the data for further analysis or machine learning. It also visualizes the results and saves them for later use.

## Main Components Explained

### 1. **preprocessing.py**

- **Loads ECG data** from CSV files, handling both single-channel and multi-channel data.
- **Normalizes** the signal (so it has zero mean and unit variance).
- **Smooths** the signal to reduce noise.
- **Detects peaks** (important points in the ECG, like R-waves) using signal processing techniques.
- **Detects both upward and downward peaks** (bipolar peak detection) for more comprehensive wave identification.

### 2. **segmentation.py**

- **Cuts the ECG signal** into smaller segments around detected peaks.
- **Labels** each segment based on its pattern (e.g., P, Q, R, S, T, U waves).
- **Extracts features** from each segment for further analysis.

### 3. **labeling.py**

- **Assigns labels** (P, Q, R, S, T, U) to each segment or peak, based on their position and characteristics.
- **Groups waves** into heartbeats (cardiac cycles) for easier analysis.
- **Annotates the signal** with per-sample wave labels for visualization and downstream processing.

### 4. **features.py**

- **Extracts numerical features** (like mean, standard deviation, max, min) from each segment or labeled wave.
- These features help describe the shape and properties of each part of the ECG.

### 5. **evolutionary.py**

- **Implements an evolutionary algorithm** to find the best way to segment the ECG signal.
- Uses random changes and selection to improve segmentation over several generations.
- **Evaluates segment fitness** using a combination of statistical and shape-based metrics.

### 6. **discretization.py**

- **Converts continuous features** into discrete bins (symbolic representation), making them easier to use for pattern recognition or machine learning.
- **Clusters features** into groups using algorithms like k-means.

### 7. **utils.py**

- **Provides helper functions** for plotting signals, segments, features, and annotations.
- **Saves results** (like features and annotations) to files for later use.
- **Plots wave annotations and segments** for visual inspection.

## How Does the Main Program Work? (main.py)

The `main.py` file is the entry point of the program. Here is what it does, step by step:

1. **Loads the ECG data** from a file.
2. **Normalizes and smooths** the signal to prepare it for analysis.
3. **Detects both upward and downward peaks** in the ECG signal.
4. **Assigns wave labels** (P, Q, R, S, T, U) to each detected peak.
5. **Plots the signal** with detected peaks and labels for visualization.
6. **Annotates the whole signal** with wave labels and saves the annotation.
7. **Extracts features** from each labeled wave or segment.
8. **Plots the extracted features** for inspection.
9. **Optionally runs an evolutionary algorithm** to find the best segmentation.
10. **Optionally clusters the features** into groups (for pattern recognition).
11. **Saves all results** (plots, features, annotations, metadata) to the output folder for later use or reporting.

## Example Data: Synthetic-three-patterns-with-noise.csv

The file `Synthetic-three-patterns-with-noise.csv` is an example ECG data file included with this project. It contains a simulated (synthetic) ECG signal with three different patterns and some added noise. Each row in the file represents a single time point in the ECG recording, and the values show the electrical activity of the heart at that moment.

This file is used to test and demonstrate what the program can do. By running the program on this example data, you can see how the different steps—such as peak detection, segmentation, labeling, feature extraction, and visualization—work in practice. It helps to check that the program is functioning correctly and to understand the results produced by each part of the pipeline.

## Summary

This program provides a complete pipeline for processing ECG signals: from loading and cleaning the data, through detecting important points and extracting features, to visualizing and saving the results. It is modular, so each part (preprocessing, segmentation, labeling, etc.) can be improved or replaced independently.
