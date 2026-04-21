#filtry, normalizacja, segmentacja sygnalu
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, low=0.5, high=40, fs=360):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')[:2]
    return filtfilt(b, a, signal)

def normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

def segment_signal(signal, window_size=360*2):
    segments = []
    for i in range(0, len(signal) - window_size, window_size):
        segments.append(signal[i:i+window_size])
    return np.array(segments)

def preprocess_pipeline(signal):
    signal = bandpass_filter(signal)
    signal = normalize(signal)
    return segment_signal(signal)
