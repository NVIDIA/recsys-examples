import torch


def translateToPowerLaw(min, max, alpha, x):
    gamma = torch.tensor([1 - alpha], device=x.device)
    y = torch.pow(
        x * (torch.pow(max, gamma) - torch.pow(min, gamma)) + torch.pow(min, gamma),
        1.0 / gamma,
    )
    b = y >= max
    y[b] = max - 1
    return y


def PowerLaw(min, max, alpha, N, device=torch.device("cuda"), permute=None):
    x = torch.rand(N, device=device, dtype=torch.float64)
    y = translateToPowerLaw(min, max, alpha, x).to(torch.int64)

    if permute is not None:
        y = permute[y]

    return y


def gen_key(batch, hotness, alpha, N, device, permute=None):
    ret = PowerLaw(1, N, alpha, hotness * batch, device, permute)
    return ret


def gen_jagged_key(
    batch,
    hotness,
    alpha,
    num_table_rows,
    device,
    feature_name,
    permute=None,
):
    """Generate a KeyedJaggedTensor with power-law distributed indices.

    Supports single-table (feature_name is str, num_table_rows is int) and
    multi-table (feature_name is List[str], num_table_rows is List[int]).
    """
    import torchrec

    if isinstance(feature_name, str):
        feature_names = [feature_name]
    else:
        feature_names = list(feature_name)

    if isinstance(num_table_rows, (int, float)):
        table_rows_list = [int(num_table_rows)] * len(feature_names)
    else:
        table_rows_list = list(num_table_rows)

    num_tables = len(feature_names)
    assert len(table_rows_list) == num_tables

    indices_list = []
    for t in range(num_tables):
        indices_list.append(
            gen_key(batch, hotness, alpha, table_rows_list[t], device, permute)
        )
    indices = torch.cat(indices_list, dim=0)
    lengths = torch.full(
        (batch * num_tables,), hotness, dtype=torch.int64, device=device
    )

    return torchrec.KeyedJaggedTensor(
        keys=feature_names,
        values=indices,
        lengths=lengths,
    )


def zipf(min_val, max_val, exponent, size, device):
    """Generates Zipf-like random variables in [min_val, max_val).

    Uses torch.multinomial on GPU instead of np.random.choice to avoid
    CPU-GPU round-trips.

    Args:
        min_val (int): Minimum value (inclusive, must be >= 0).
        max_val (int): Maximum value (exclusive).
        exponent (float): Exponent parameter (a > 0).
        size (int): Number of samples.
        device: Target torch device.

    Returns:
        torch.Tensor: Sampled values of specified size on *device*.
    """
    n = max_val - min_val
    values = torch.arange(1, n + 1, dtype=torch.float64, device=device)
    probs = 1.0 / (values**exponent)
    probs = (probs / probs.sum()).float()

    k = torch.arange(min_val, max_val, dtype=torch.long, device=device)
    perm = torch.randperm(k.size(0), device=device)
    k_shuffled = k[perm]

    sample_indices = torch.multinomial(probs, size, replacement=True)
    samples = k_shuffled[sample_indices]
    return samples


if __name__ == "__main__":
    zipf(0, 100, 1.05, 100, torch.device("cuda:0"))
    zipf(0, 100, 1.2, 100, torch.device("cuda:0"))
