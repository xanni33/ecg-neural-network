import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_signal(signal, title="example ECG"):
    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("samples")
    plt.ylabel("mV")
    plt.show()

def plot_signal_length_distribution(signals):
    lengths = [len(s) for s in signals]
    plt.hist(lengths, bins=50)
    plt.title("distribution of signal lengths")
    plt.xlabel("length")
    plt.ylabel("count")
    plt.show()

def plot_label_histogram(label_lists):
    all_labels = []
    for lbls in label_lists:
        all_labels.extend(lbls)

    counter = Counter(all_labels)
    plt.bar(counter.keys(), counter.values())
    plt.title("label distribution")
    plt.xlabel("class")
    plt.ylabel("count")
    plt.show()
