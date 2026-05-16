import numpy as np
from scipy.signal import butter, filtfilt

# -----------------------------
# 1. NOISE (Gaussian noise)
# -----------------------------
def add_gaussian_noise(signal, noise_factor=0.01):
    noise = np.random.normal(0, 1, len(signal))
    return signal + noise_factor * noise


# -----------------------------
# 2. TIME SHIFT
# -----------------------------
def time_shift(signal, shift_max=50):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(signal, shift)


# -----------------------------
# 3. SCALING (amplitude)
# -----------------------------
def scaling(signal, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return signal * scale


# -----------------------------
# 4. BANDPASS (stabilizacja po augmentacji)
# -----------------------------
def bandpass_filter(signal, low=0.5, high=40, fs=100):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, signal)


# -----------------------------
# 5. PIPELINE AUGMENTATION
# -----------------------------
def augment_signal(signal, p=0.7):
    """
    Applies augmentation with probability p.
    """
    if np.random.rand() > p:
        return signal

    # random sequence of augmentations
    if np.random.rand() < 0.5:
        signal = add_gaussian_noise(signal)

    if np.random.rand() < 0.5:
        signal = time_shift(signal)

    if np.random.rand() < 0.5:
        signal = scaling(signal)

    # always stabilize
    signal = bandpass_filter(signal)

    return signal

#to-do teraz: zmodyfikować preprocessing by augmentacja byla elementem pipeline; zmiana generowania trainset
#w notebooku treningowym; final eval; tabela wynikow z augmentacja vs bez; analiza bledow; wnioski