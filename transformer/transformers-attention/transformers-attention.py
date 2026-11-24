import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    d_k = K.shape[-1]
    q_kt = torch.matmul(Q, K.transpose(-2, -1))

    scores = torch.softmax(q_kt/math.sqrt(d_k), dim=-1)
    
    return  torch.matmul(scores, V)