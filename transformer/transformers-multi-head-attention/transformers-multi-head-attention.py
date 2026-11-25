import numpy as np
import torch

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q = torch.as_tensor(Q, dtype=torch.float32)
    K = torch.as_tensor(K, dtype=torch.float32)
    V = torch.as_tensor(V, dtype=torch.float32)
    
    W_q = torch.as_tensor(W_q, dtype=torch.float32)
    W_k = torch.as_tensor(W_k, dtype=torch.float32)
    W_v = torch.as_tensor(W_v, dtype=torch.float32)
    W_o = torch.as_tensor(W_o, dtype=torch.float32)

    q_w = torch.matmul(Q, W_q)
    k_w = torch.matmul(K, W_k)
    v_w = torch.matmul(V, W_v)

    q = q_w.reshape(batch_size, seq_len, num_heads, d_k)
    k = k_w.reshape(batch_size, seq_len, num_heads, d_k)
    v = v_w.reshape(batch_size, seq_len, num_heads, d_k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    k_t = k.transpose(2, 3)

    q_k_t = torch.matmul(q, k_t)
    scores = q_k_t / np.sqrt(d_k)

    scores = torch.softmax(scores, dim=-1)

    attention = torch.matmul(scores, v)

    attention = attention.transpose(1, 2)
    attention = attention.reshape(batch_size, seq_len, d_model)

    output = torch.matmul(attention, W_o)

    return output.numpy()