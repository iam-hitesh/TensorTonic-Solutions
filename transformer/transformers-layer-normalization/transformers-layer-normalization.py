import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """

    x_mean = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    var_sqrt = np.sqrt(variance + eps)

    res = x_mean / var_sqrt

    res = gamma * res
    res = res + beta

    return res