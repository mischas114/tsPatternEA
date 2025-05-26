import numpy as np

def match_wave_type(segment):
    """
    Return a known wave label (P,Q,R,S,T,U) if segment matches a pattern,
    otherwise return an empty string.
    """
    # Example domain logic: checks amplitude threshold for 'R'
    # (in practice, compare shape or correlation, etc.)
    if np.max(segment) > 0.8:
        return 'R'
    # ...additional logic for other wave types...
    return ''

def assign_labels(segments, peaks=None, r_peak_indices=None):
    # Improved: Assign P, Q, R, S, T based on position relative to R-peak
    # If r_peak_indices is provided, use it; else, use all peaks as R
    labels = [''] * len(segments)
    if r_peak_indices is None:
        # Assume all peaks are R-peaks
        r_peak_indices = list(range(len(segments)))
    for i, r_idx in enumerate(r_peak_indices):
        # Assign Q and S to neighbors if possible
        if r_idx > 0:
            labels[r_idx - 1] = 'Q'
        labels[r_idx] = 'R'
        if r_idx < len(segments) - 1:
            labels[r_idx + 1] = 'S'
        # Assign P before Q, T after S if possible
        if r_idx > 1:
            labels[r_idx - 2] = 'P'
        if r_idx < len(segments) - 2:
            labels[r_idx + 2] = 'T'
    return labels

def annotate_waves(signal, peaks):
    from segmentation import extract_segments  # moved import here to avoid circular import
    segments = extract_segments(signal, peaks)
    # Assume all peaks are R-peaks for now
    labels = assign_labels(segments)
    label_array = [''] * len(signal)
    for p, label in zip(peaks, labels):
        if 0 <= p < len(signal):
            label_array[p] = label
    return label_array

def group_waves_into_heartbeats(peaks, labels, min_rr=100):
    """
    Group detected waves into heartbeats (cardiac cycles) based on R-peak positions.
    Returns a list of dicts: [{"R": idx, "Q": idx, ...}, ...]
    """
    heartbeats = []
    current_beat = {}
    for i, (p, lab) in enumerate(zip(peaks, labels)):
        if lab == 'R':
            if current_beat:
                heartbeats.append(current_beat)
            current_beat = {'R': p}
        elif lab in {'P', 'Q', 'S', 'T', 'U'}:
            current_beat[lab] = p
    if current_beat:
        heartbeats.append(current_beat)
    return heartbeats