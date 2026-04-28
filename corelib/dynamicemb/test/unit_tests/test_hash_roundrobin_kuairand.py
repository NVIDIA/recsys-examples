import csv
import os
from collections import Counter

import numpy as np
import pytest
import torch


HASH_SIZE = 10_000_000
KUAIRAND_1K_FEATURES = {
    "video_id": 49332,
    "user_id": 1000,
}


def hash_key_cpu(keys_np: np.ndarray) -> np.ndarray:
    k = np.asarray(keys_np, dtype=np.uint64)
    k = k ^ (k >> np.uint64(33))
    k = k * np.uint64(0xFF51AFD7ED558CCD)
    k = k ^ (k >> np.uint64(33))
    k = k * np.uint64(0xC4CEB9FE1A85EC53)
    k = k ^ (k >> np.uint64(33))
    return k


def assign_owner_cpu(
    indices: np.ndarray,
    my_size: int,
    dist_type: str,
    blk_size: int,
) -> np.ndarray:
    if dist_type == "continuous":
        return np.where(
            indices < blk_size * my_size,
            indices // blk_size,
            indices % my_size,
        ).astype(np.int64)
    if dist_type == "roundrobin":
        return (indices % my_size).astype(np.int64)
    if dist_type == "hash_roundrobin":
        return (hash_key_cpu(indices) % np.uint64(my_size)).astype(np.int64)
    raise ValueError(f"Unsupported dist_type: {dist_type}")


def owner_histogram_cpu(
    indices: np.ndarray,
    my_size: int,
    dist_type: str,
    blk_size: int,
) -> np.ndarray:
    owners = assign_owner_cpu(indices, my_size, dist_type, blk_size)
    return np.bincount(owners, minlength=my_size)


def imbalance_ratio(hist: np.ndarray) -> float:
    expected = hist.sum() / len(hist)
    return float((hist.max() - hist.min()) / expected)


def max_min_ratio(hist: np.ndarray) -> float:
    return float(hist.max() / max(hist.min(), 1))


def try_load_kuairand_indices():
    possible_paths = [
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "examples",
            "hstu",
            "training",
            "data",
            "KuaiRand-1K",
            "data",
        ),
        os.path.join(
            os.environ.get("KUAIRAND_DATA_PATH", ""),
            "KuaiRand-1K",
            "data",
        ),
    ]

    for base_path in possible_paths:
        log_files = [
            os.path.join(base_path, "log_standard_4_08_to_4_21_1k.csv"),
            os.path.join(base_path, "log_standard_4_22_to_5_08_1k.csv"),
        ]
        if all(os.path.exists(f) for f in log_files):
            all_video_ids = []
            all_user_ids = []
            for log_file in log_files:
                with open(log_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_user_ids.append(int(row["user_id"]))
                        all_video_ids.append(int(row["video_id"]))
            return {
                "user_id": np.array(all_user_ids, dtype=np.int64),
                "video_id": np.array(all_video_ids, dtype=np.int64),
            }
    return None


def generate_zipf_logical_indices(
    feature_name: str,
    num_interactions: int = 4_369_953,
) -> np.ndarray:
    np.random.seed(42)
    num_unique = KUAIRAND_1K_FEATURES[feature_name]
    exponent = 1.5 if feature_name == "user_id" else 1.8

    ranks = np.arange(1, num_unique + 1)
    probs = 1.0 / (ranks**exponent)
    probs /= probs.sum()
    return np.random.choice(ranks, size=num_interactions, replace=True, p=probs).astype(
        np.int64
    )


def make_adversarial_modulo_aliasing_keys(
    logical_ids: np.ndarray,
    my_size: int,
    residue: int = 0,
) -> np.ndarray:
    logical_ids = np.asarray(logical_ids, dtype=np.int64)
    if my_size <= 0:
        raise ValueError(f"my_size must be positive, got {my_size}")
    if residue < 0:
        raise ValueError(f"residue must be non-negative, got {residue}")

    max_int64 = np.iinfo(np.int64).max
    max_logical_id = (max_int64 - residue) // my_size
    if logical_ids.size > 0 and logical_ids.max() > max_logical_id:
        raise OverflowError(
            f"adversarial remap would overflow int64: max logical id "
            f"{logical_ids.max()} > {max_logical_id}"
        )

    return logical_ids * my_size + residue


def remap_frequency_histogram_to_adversarial_keys(
    logical_indices: np.ndarray,
    my_size: int,
    residue: int = 0,
) -> np.ndarray:
    return make_adversarial_modulo_aliasing_keys(logical_indices, my_size, residue)


def print_distribution_summary(title: str, hist: np.ndarray) -> None:
    print(title)
    print(f"  Distribution: {hist.tolist()}")
    print(f"  Imbalance: {imbalance_ratio(hist):.2%}")
    print(f"  Max/Min: {max_min_ratio(hist):.2f}x")


@pytest.mark.parametrize("my_size", [4, 8])
def test_hash_roundrobin_breaks_modulo_aliasing_cpu(my_size):
    logical_ids = np.arange(50_000, dtype=np.int64)
    indices = make_adversarial_modulo_aliasing_keys(logical_ids, my_size, residue=0)
    blk_size = HASH_SIZE // my_size

    rr_hist = owner_histogram_cpu(indices, my_size, "roundrobin", blk_size)
    hr_hist = owner_histogram_cpu(indices, my_size, "hash_roundrobin", blk_size)

    print_distribution_summary(f"\nroundrobin aliasing (my_size={my_size})", rr_hist)
    print_distribution_summary(
        f"\nhash_roundrobin aliasing (my_size={my_size})", hr_hist
    )

    assert rr_hist.max() == len(indices)
    assert rr_hist.min() == 0
    assert imbalance_ratio(hr_hist) < 0.30
    assert max_min_ratio(hr_hist) < 1.50


@pytest.mark.parametrize("feature_name", ["user_id", "video_id"])
@pytest.mark.parametrize("my_size", [4, 8])
def test_hash_roundrobin_is_pattern_robust_under_same_logical_load(
    feature_name,
    my_size,
):
    logical_indices = generate_zipf_logical_indices(feature_name)
    blk_size = HASH_SIZE // my_size

    natural_indices = logical_indices.copy()
    adversarial_indices = remap_frequency_histogram_to_adversarial_keys(
        logical_indices,
        my_size,
        residue=0,
    )

    natural_rr = owner_histogram_cpu(natural_indices, my_size, "roundrobin", blk_size)
    adversarial_rr = owner_histogram_cpu(
        adversarial_indices,
        my_size,
        "roundrobin",
        blk_size,
    )
    natural_hr = owner_histogram_cpu(
        natural_indices,
        my_size,
        "hash_roundrobin",
        blk_size,
    )
    adversarial_hr = owner_histogram_cpu(
        adversarial_indices,
        my_size,
        "hash_roundrobin",
        blk_size,
    )

    print_distribution_summary(
        f"\n{feature_name} natural roundrobin (my_size={my_size})",
        natural_rr,
    )
    print_distribution_summary(
        f"\n{feature_name} adversarial roundrobin (my_size={my_size})",
        adversarial_rr,
    )
    print_distribution_summary(
        f"\n{feature_name} natural hash_roundrobin (my_size={my_size})",
        natural_hr,
    )
    print_distribution_summary(
        f"\n{feature_name} adversarial hash_roundrobin (my_size={my_size})",
        adversarial_hr,
    )

    rr_natural_imb = imbalance_ratio(natural_rr)
    rr_adversarial_imb = imbalance_ratio(adversarial_rr)
    hr_natural_imb = imbalance_ratio(natural_hr)
    hr_adversarial_imb = imbalance_ratio(adversarial_hr)

    rr_imb_increase = rr_adversarial_imb - rr_natural_imb
    hr_imb_increase = hr_adversarial_imb - hr_natural_imb

    assert rr_adversarial_imb > rr_natural_imb + 0.50
    assert hr_adversarial_imb < rr_adversarial_imb
    assert hr_imb_increase < rr_imb_increase


@pytest.mark.parametrize("feature_name", ["user_id", "video_id"])
def test_key_frequency_skew(feature_name):
    real_data = try_load_kuairand_indices()
    if real_data is None:
        indices = generate_zipf_logical_indices(feature_name)
    else:
        indices = real_data[feature_name]

    freq = Counter(indices.tolist())
    counts = np.array(sorted(freq.values(), reverse=True))

    total = counts.sum()
    top1_pct = counts[0] / total * 100
    top10_pct = counts[:10].sum() / total * 100
    top100_pct = counts[:100].sum() / total * 100
    gini = 1 - 2 * np.trapz(
        np.cumsum(sorted(counts)) / total,
        np.linspace(0, 1, len(counts)),
    )

    print(f"\n=== Key Frequency Skew: {feature_name} ===")
    print(f"Unique keys: {len(counts)}")
    print(f"Top-1 key:   {top1_pct:.2f}% of all interactions")
    print(f"Top-10 keys: {top10_pct:.2f}% of all interactions")
    print(f"Top-100 keys:{top100_pct:.2f}% of all interactions")
    print(f"Gini coefficient: {gini:.4f}")

    assert gini > 0.3


@pytest.mark.parametrize("feature_name", ["user_id", "video_id"])
@pytest.mark.parametrize("my_size", [4, 8])
def test_real_or_zipf_histogram_with_adversarial_remap(feature_name, my_size):
    real_data = try_load_kuairand_indices()
    if real_data is None:
        logical_indices = generate_zipf_logical_indices(feature_name)
    else:
        logical_indices = real_data[feature_name]

    adversarial_indices = remap_frequency_histogram_to_adversarial_keys(
        logical_indices,
        my_size,
        residue=0,
    )
    blk_size = HASH_SIZE // my_size

    rr_hist = owner_histogram_cpu(
        adversarial_indices,
        my_size,
        "roundrobin",
        blk_size,
    )
    hr_hist = owner_histogram_cpu(
        adversarial_indices,
        my_size,
        "hash_roundrobin",
        blk_size,
    )

    print_distribution_summary(
        f"\n{feature_name} adversarial-remap roundrobin (my_size={my_size})",
        rr_hist,
    )
    print_distribution_summary(
        f"\n{feature_name} adversarial-remap hash_roundrobin (my_size={my_size})",
        hr_hist,
    )

    assert imbalance_ratio(hr_hist) < imbalance_ratio(rr_hist)
    assert max_min_ratio(hr_hist) < max_min_ratio(rr_hist)


@pytest.mark.parametrize("feature_name", ["user_id", "video_id"])
@pytest.mark.parametrize("my_size", [4, 8])
def test_bucketize_kernel_matches_cpu_reference(feature_name, my_size):
    try:
        from dynamicemb_extensions import block_bucketize_sparse_features
    except ImportError:
        pytest.skip("dynamicemb_extensions not available")

    logical_indices = generate_zipf_logical_indices(feature_name, num_interactions=200_000)
    adversarial_indices = remap_frequency_histogram_to_adversarial_keys(
        logical_indices,
        my_size,
        residue=0,
    )

    batch_size = min(50_000, len(adversarial_indices))
    sample_idx = np.random.choice(
        len(adversarial_indices),
        size=batch_size,
        replace=False,
    )
    batch_indices = adversarial_indices[sample_idx].astype(np.int64)

    indices = torch.tensor(batch_indices, dtype=torch.int64, device="cuda")
    lengths = torch.tensor([batch_size], dtype=torch.int64, device="cuda")
    block_sizes = torch.tensor(
        [HASH_SIZE // my_size],
        dtype=torch.int64,
        device="cuda",
    )

    for dist_type_val, dist_type_name in [
        (1, "roundrobin"),
        (2, "hash_roundrobin"),
    ]:
        dist_type_per_feature = torch.tensor(
            [dist_type_val],
            dtype=torch.int32,
            device="cuda",
        )

        result = block_bucketize_sparse_features(
            lengths,
            indices,
            bucketize_pos=False,
            sequence=False,
            dist_type_per_feature=dist_type_per_feature,
            block_sizes=block_sizes,
            my_size=my_size,
            max_B=1,
        )

        new_lengths = result[0].cpu().numpy()
        new_indices = result[1]
        rank_counts = new_lengths[:my_size]
        total_output = rank_counts.sum()

        cpu_hist = owner_histogram_cpu(
            batch_indices,
            my_size,
            dist_type_name,
            HASH_SIZE // my_size,
        )

        print(f"\n{dist_type_name} kernel (my_size={my_size}, {feature_name}):")
        print(f"  CUDA per-rank counts: {rank_counts.tolist()}")
        print(f"  CPU  per-rank counts: {cpu_hist.tolist()}")

        assert total_output == batch_size
        assert torch.all(new_indices >= 0).item()
        assert rank_counts.tolist() == cpu_hist.tolist()

