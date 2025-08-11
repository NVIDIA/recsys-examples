
import torch
import numpy as np

def translateToPowerLaw(min, max, alpha, x):
    gamma = torch.tensor([1 - alpha], device=x.device)
    y = torch.pow(x * (torch.pow(max, gamma) - torch.pow(min, gamma)) + torch.pow(min, gamma),
          1.0 / gamma)
    b = (y >= max)
    y[b] = max-1
    return y

def PowerLaw(min, max, alpha, N, device=torch.device('cuda'), permute=None):
    x = torch.rand(N, device=device, dtype=torch.float64)
    y = translateToPowerLaw(min, max, alpha, x).to(torch.int64)

    if permute != None:
        y = permute[y]

    return y

def gen_key(batch, hotness, alpha, N, device, permute=None):
    ret = PowerLaw(1, N, alpha, hotness*batch, device, permute)
    return ret

def gen_jagged_key(batch, hotness, alpha, num_table_rows, device, feature_name, permute=None):
    import torchrec
    key = gen_key(batch, hotness, alpha, num_table_rows, device, permute)
    lengths = torch.tensor([hotness]*batch, dtype=torch.int64, device=device)
    return torchrec.KeyedJaggedTensor(
        keys=[feature_name],
        values=key,
        lengths=lengths,
    )

def zipf(min_val, max_val, exponent, size, device):
    """
    Generates Zipf-like random variables in the inclusive range [min_val, max_val).

    Args:
        min_val (int): Minimum value (inclusive, must be ≥0).
        max_val (int): Maximum value (exclusive).
        exponent (float): Exponent parameter (a > 0).
        size (int): Output shape.

    Returns:
        torch.Tensor: Sampled values of specified size.
    """

    # Generate integer values and probabilities
    values = torch.arange(min_val + 1, max_val + 1, dtype=torch.long, device=device)
    probs = 1.0 / (values.float() ** exponent)
    probs_normalized = probs / probs.sum()
    print(f"Exponent:{exponent}")
    for x in probs_normalized:
      print(x.item())

    k = torch.arange(min_val, max_val, dtype=torch.long, device=device)
    perm = torch.randperm(k.size(0), device=device)
    k_shuffled = k[perm]
    print(f"Permute:")
    for x in k_shuffled:
      print(x.item()) 

    probs_np = probs_normalized.cpu().numpy()
    samples = np.random.choice(k_shuffled.cpu().numpy(), size=size, replace=True, p=probs_np)
    samples = torch.tensor(samples, device=probs_normalized.device)
    print(f"Result:")
    for x in samples:
      print(x.item())    
    

if __name__ == "__main__":
  zipf(0, 100, 1.05, 100, torch.device("cuda:0"))
  zipf(0, 100, 1.2, 100, torch.device("cuda:0"))