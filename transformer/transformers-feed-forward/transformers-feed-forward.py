import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward output.
    """
    x_w = np.matmul(x, W1)
    relu = np.maximum(0, x_w + b1)

    relu_w2 = np.matmul(relu, W2)

    out = relu_w2 + b2

    return out