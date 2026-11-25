import numpy as np
import torch

def softmax(x, axis=-1):
    """Provided: numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns x normalized over the last axis, then scaled by gamma and shifted by beta.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    
    variance = np.var(x, axis=-1, keepdims=True)
    normalized = centered / np.sqrt(variance + eps)
    
    return gamma * normalized + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns multi-head attention of Q, K, V, projecting them with W_q, W_k, W_v and the merged heads with W_o.
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

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward network output with a ReLU hidden layer.
    """
    x_w = np.matmul(x, W1)
    relu = np.maximum(0, x_w + b1)

    relu_w2 = np.matmul(relu, W2)

    out = relu_w2 + b2

    return out

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    mha = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x1 = layer_norm(
        x + mha, gamma1, beta1
    )
    ffn = feed_forward(
        x1, W1, b1, W2, b2
    )

    out = layer_norm(
        x1 + ffn, gamma2, beta2
    )

    return out
