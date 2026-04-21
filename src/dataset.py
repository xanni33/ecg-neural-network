from sklearn.model_selection import train_test_split
import numpy as np
import os

def prepare_dataset(segments, labels, test_size=0.2):
    """
    prosty split datasetu (single-label, bez stratyfikacji)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        segments,
        labels,
        test_size=test_size,
        random_state=42,
        shuffle=True
    )

    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_test.npy", y_test)

    print("saved dataset:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)


def prepare_multilabel_dataset(segments, labels, test_size=0.2):
    """
    dataset split dla multi-label (bez stratify - sklearn tego nie obsluguje)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        segments,
        labels,
        test_size=test_size,
        random_state=42,
        shuffle=True
    )

    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/X_train_ml.npy", X_train)
    np.save("data/processed/X_test_ml.npy", X_test)
    np.save("data/processed/y_train_ml.npy", y_train)
    np.save("data/processed/y_test_ml.npy", y_test)

    print("saved MULTI-LABEL dataset:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
