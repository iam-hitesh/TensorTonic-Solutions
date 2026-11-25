import numpy as np
import torch

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    B, S_q, d_model = Q.shape
    S_k = K.shape[1]
    d_k = d_model // num_heads

    Q, K, V, W_q, W_k, W_v, W_o = (torch.as_tensor(x, dtype=torch.float32)
                                   for x in (Q, K, V, W_q, W_k, W_v, W_o))

    with torch.no_grad():
        q = (Q @ W_q).reshape(B, S_q, num_heads, d_k).transpose(1, 2)
        k = (K @ W_k).reshape(B, S_k, num_heads, d_k).transpose(1, 2)
        v = (V @ W_v).reshape(B, S_k, num_heads, d_k).transpose(1, 2)

        scores = (q @ k.transpose(2, 3)) / d_k ** 0.5
        weights = torch.softmax(scores, dim=-1)

        out = weights @ v                                  # (B, H, S_q, d_k)
        out = out.transpose(1, 2).reshape(B, S_q, d_model)
        out = out @ W_o

    return out.numpy()