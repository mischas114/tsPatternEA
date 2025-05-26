import wfdb
import pandas as pd
import os

# Path to your data folder
base_path = ".convert-WFDB"
record_name = "100"
full_path = os.path.join(base_path, record_name)

# Load the ECG signal and annotations
record = wfdb.rdrecord(full_path)       # loads 100.dat and 100.hea
annotation = wfdb.rdann(full_path, 'atr')  # loads 100.atr

# Convert signal to DataFrame
signal = pd.DataFrame(record.p_signal, columns=record.sig_name)
signal['Time'] = [i / record.fs for i in range(len(signal))]

# Save signal to CSV
signal.to_csv('ecg_100_signal.csv', index=False)

# Save annotations separately if needed
ann_df = pd.DataFrame({
    'Sample': annotation.sample,
    'Time': [s / record.fs for s in annotation.sample],
    'Symbol': annotation.symbol
})
ann_df.to_csv(base_path,'ecg_100_annotations.csv', index=False)
