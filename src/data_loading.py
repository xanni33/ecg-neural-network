# src/data_loading.py
import wfdb
import numpy as np
import os

# teraz wskazujemy dokładnie folder z plikami .hea/.dat
DATASET_PATH = os.path.join("data", "raw", "ptbxl")


def load_record(record_name):
    """load ECG record using WFDB."""
    record = wfdb.rdrecord(os.path.join(DATASET_PATH, record_name))
    annotation = wfdb.rdann(os.path.join(DATASET_PATH, record_name), 'atr')
    return record.p_signal[:, 0], annotation.symbol, annotation.sample

def load_all_records(record_list):
    signals = []
    labels = []
    samples = []

    for rec in record_list:
        print(f"loading {rec}...")
        sig, ann, sam = load_record(rec)
        signals.append(sig)
        labels.append(ann)
        samples.append(sam)

    return signals, labels, samples
