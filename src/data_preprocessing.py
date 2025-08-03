import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_and_preprocess(sample_size=5000, test_size=0.2, random_state=42):
    """
    Fetch MNIST, sample a subset, normalize, reshape, and split.
    Returns X_train, X_test, y_train, y_test.
    """
    # Fetch full MNIST
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist['data'], mnist['target'].astype(int)

    # Sample subset
    idx = np.random.RandomState(random_state).choice(
        np.arange(X.shape[0]), sample_size, replace=False
    )
    X, y = X[idx], y[idx]

    # Handle missing (just in case)
    X = pd.DataFrame(X).fillna(0).values

    # Normalize to [0,1]
    X = X / 255.0

    # No reshape needed: already (n_samples, 784)

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
