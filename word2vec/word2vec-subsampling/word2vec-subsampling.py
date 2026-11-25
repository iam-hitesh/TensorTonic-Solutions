import torch

def subsample_keep_probs(counts: torch.Tensor,
                         t: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 keep probability for every vocabulary word.
    """
    # 1. Calculate total corpus count
    total_count = torch.sum(counts)

    # 2. Convert counts into relative frequencies
    f_w = counts / total_count

    # 3. Calculate keep probabilities
    keep_probs = torch.sqrt(t / f_w)

    # 4. Clamp probabilities to maximum 1
    keep_probs = torch.clamp(keep_probs, max=1.0)

    # 5. Return float64 tensor
    return keep_probs.to(torch.float64)