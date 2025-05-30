import numpy as np
from typing import List

# ======================================================================
# ECG peak → canonical‑wave labelling (P/Q/R/S/T/U) + heartbeat counter
# ======================================================================
# The mission of this module is *only* to assign a **letter + beat index**
# to every peak location you feed in.  It does *not* do peak detection –
# the outer pipeline must give us a **comprehensive** list of upward *and*
# downward extrema (see the note in README below).
#
# Example output for a 2‑beat strip               R‑R interval (samples)
# ──────────────────────────────────────────┐              ┌───────────────┐
#   P1  Q1  R1  S1  T1   P2  Q2  R2  S2  T2             beat_idx = 1, 2…
#   ↑   ↑   ↑   ↑   ↑    ↑   ↑   ↑   ↑   ↑
#   |   |   |   |   |    |   |   |   |   |
#   letter per peak (this file) ──────────┘
#
# Heuristics
# ----------
# 1. R‑waves → tallest *positive* peaks (amplitude > rel_thresh · max(|x|)).
# 2. Relatives (Q,S) are the first *negative* peaks just before / after R.
# 3. P and T are lower‑amplitude *positive* peaks further out in time.
# 4. Any peak that does not match a rule keeps its raw amplitude string so
#    you can still visualise it on a plot.
#
# Feel free to tune the constants for your dataset – they are collected at
# the top of the file.
# ======================================================================

DEFAULT_FS = 250           # Hz (adjust to your recorder)
R_THRESH   = 0.6           # fraction of global max() → R detection
MIN_PQR_STRENGTH = 0.1     # fraction of max(|signal|) for P/Q/S/T eligibility

# Time‑window rules expressed as *fractions of the surrounding RR interval*
WINDOWS = {
    "P": (-0.40, -0.10),   # before R, positive
    "Q": (-0.12, -0.02),   # just before R, negative
    "R": (-0.02,  0.02),   # central, positive & tallest
    "S": ( 0.02,  0.12),   # just after R, negative
    "T": ( 0.12,  0.60),   # after S, positive
    "U": ( 0.60,  0.85),   # optional late positive bump
}

# ----------------------------------------------------------------------
# Helper – find R peaks (tall positive deflections)
# ----------------------------------------------------------------------

def detect_r_peaks(signal: np.ndarray, peaks: List[int], rel_threshold: float = R_THRESH):
    """Return indices from *peaks* that most likely are R‑waves."""
    abs_max = np.max(np.abs(signal))
    if abs_max == 0:
        return []
    thr = rel_threshold * abs_max
    return [p for p in peaks if signal[p] > thr]

# ----------------------------------------------------------------------
# Main classifer
# ----------------------------------------------------------------------

def classify_peak(rel_pos: float, amp: float, global_max: float) -> str:
    """Map one peak into a P/Q/R/S/T/U letter (or '' if unknown)."""
    strength = np.abs(amp) / global_max

    for letter, (lo, hi) in WINDOWS.items():
        if lo <= rel_pos <= hi:
            # sign check – Q & S must be *negative*, others *positive*
            if letter in ("Q", "S") and amp < -MIN_PQR_STRENGTH * global_max:
                return letter
            if letter in ("P", "R", "T", "U") and amp > MIN_PQR_STRENGTH * global_max:
                return letter
    return ""

# ----------------------------------------------------------------------
# assign_labels – public API
# ----------------------------------------------------------------------

def assign_labels(signal: np.ndarray, peaks: List[int], fs: int = DEFAULT_FS):
    """Return array of strings, one per peak: P1, Q1, R1, ..."""
    if len(peaks) == 0:
        return np.array([])

    global_max = np.max(np.abs(signal))
    r_peaks = detect_r_peaks(signal, peaks)
    if len(r_peaks) == 0:
        return np.array([f"{signal[p]:.2f}" for p in peaks])  # give up

    labels = []
    for p in peaks:
        # ---------- find neighbouring R’s to get current RR interval -----
        nearest_r_idx = np.argmin([abs(p - r) for r in r_peaks])
        nearest_r = r_peaks[nearest_r_idx]
        beat_no   = nearest_r_idx + 1                # 1‑based

        # RR interval in samples ------------------------------------------------
        if nearest_r_idx + 1 < len(r_peaks):
            rr = r_peaks[nearest_r_idx + 1] - nearest_r
        elif nearest_r_idx > 0:
            rr = nearest_r - r_peaks[nearest_r_idx - 1]
        else:
            rr = int(0.8 * fs)
        if rr == 0:
            rr = 1

        rel_pos = (p - nearest_r) / rr              # −0.5 … +0.5 …
        letter  = classify_peak(rel_pos, signal[p], global_max)

        if letter == "":
            # Unknown → keep raw amplitude (so it shows in plots)
            letter = f"{signal[p]:.2f}"

        labels.append(f"{letter}{beat_no}")

    return np.array(labels)

# ----------------------------------------------------------------------
#   Convenience helpers -------------------------------------------------
# ----------------------------------------------------------------------

def annotate_waves(signal: np.ndarray, peaks: List[int], fs: int = DEFAULT_FS):
    """Per‑sample label track ('' outside peaks)."""
    labels_per_peak = assign_labels(signal, peaks, fs)
    track = [''] * len(signal)
    for p, lab in zip(peaks, labels_per_peak):
        if 0 <= p < len(signal):
            track[p] = lab
    return track


def group_waves_into_heartbeats(peaks: List[int], labels_per_peak):
    """Return list of dicts – one per heartbeat."""
    beats = []
    for p_idx, lab in zip(peaks, labels_per_peak):
        if not lab:
            continue
        letter = ''.join(filter(str.isalpha, lab))
        beat_no = int(''.join(filter(str.isdigit, lab))) - 1
        while len(beats) <= beat_no:
            beats.append({})
        beats[beat_no][letter] = p_idx
    return beats