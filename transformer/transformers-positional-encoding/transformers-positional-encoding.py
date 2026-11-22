import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    positions = np.arange(seq_length)[:, np.newaxis]
    dimensions = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (dimensions // 2)) / d_model
    )

    angle_rads = positions * angle_rates

    sin = np.sin(angle_rads)
    cos = np.cos(angle_rads)

    pos_encoding = np.where(
        dimensions % 2 == 0,
        sin,
        cos
    )

    return pos_encoding