import csv
import os
from collections import Counter
from typing import Tuple

import numpy as np
import pytest
import torch

HASH_SIZE = 10_000_000
KUAIRAND_1K_FEATURES = {
    "video_id": 49332,
    "user_id": 1000,
}


def hash_key_cpu(keys_np: np.ndarray) -> np.ndarray:
    """Deliberately a second implementation, not an import.

    This oracle exists to check the CUDA bucketize kernel's key->rank mapping.
    Calling the production ``murmur3_fmix64`` here would compare that host code
    against itself and stop testing the device at all, so the duplication is the
    point -- do not "unify" it away.
    """
    k = np.asarray(keys_np, dtype=np.uint64)
    k = k ^ (k >> np.uint64(33))
    k = k * np.uint64(0xFF51AFD7ED558CCD)
    k = k ^ (k >> np.uint64(33))
    k = k * np.uint64(0xC4CEB9FE1A85EC53)
    k = k ^ (k >> np.uint64(33))
    return k


def stored_keys_cpu(indices: np.ndarray) -> np.ndarray:
    """The keys as the unsigned 64-bit numbers roundrobin and hash_roundrobin read.

    A table stores a key widened to int64 with its sign, so an int32 key and the
    int64 key with the same value are one key, with one rank. The rules read
    those 64 bits as unsigned, so a negative key counts as a large number.
    """
    return np.asarray(indices).astype(np.int64).astype(np.uint64)


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
        return (stored_keys_cpu(indices) % np.uint64(my_size)).astype(np.int64)
    if dist_type == "hash_roundrobin":
        hashes = hash_key_cpu(stored_keys_cpu(indices))
        return (hashes % np.uint64(my_size)).astype(np.int64)
    raise ValueError(f"Unsupported dist_type: {dist_type}")


def continuous_owner_cpu(
    indices: np.ndarray,
    key_dtype: torch.dtype,
    my_size: int,
    blk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank and rewritten key of each key under ``continuous``.

    This rule reads a key in its own width, unsigned, and writes the rewritten
    key back in that width: an int32 key stays an int32. So a negative key
    counts as a large number past the last block, and goes by the modulo.
    """
    unsigned_dtype = np.uint32 if key_dtype == torch.int32 else np.uint64
    signed_dtype = np.int32 if key_dtype == torch.int32 else np.int64
    keys = np.asarray(indices).astype(signed_dtype).astype(unsigned_dtype)
    block = unsigned_dtype(blk_size)
    size = unsigned_dtype(my_size)
    in_blocks = keys < block * size
    ranks = np.where(in_blocks, keys // block, keys % size)
    rewritten = np.where(in_blocks, keys % block, keys // size)
    return ranks.astype(np.int64), rewritten.astype(signed_dtype)


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


@pytest.mark.parametrize("key_dtype", [torch.int64, torch.int32], ids=str)
@pytest.mark.parametrize("feature_name", ["user_id", "video_id"])
@pytest.mark.parametrize("my_size", [3, 4, 7, 8])
def test_bucketize_kernel_matches_cpu_reference(feature_name, my_size, key_dtype):
    try:
        from dynamicemb_extensions import block_bucketize_sparse_features
    except ImportError:
        pytest.skip("dynamicemb_extensions not available")

    logical_indices = generate_zipf_logical_indices(
        feature_name, num_interactions=200_000
    )
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

    indices = torch.tensor(batch_indices, dtype=key_dtype, device="cuda")
    lengths = torch.tensor([batch_size], dtype=torch.int64, device="cuda")
    # In the keys' dtype, as bucketize_kjt_before_all2all passes them.
    block_sizes = torch.tensor(
        [HASH_SIZE // my_size],
        dtype=key_dtype,
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

        print(
            f"\n{dist_type_name} kernel "
            f"(my_size={my_size}, {feature_name}, {key_dtype}):"
        )
        print(f"  CUDA per-rank counts: {rank_counts.tolist()}")
        print(f"  CPU  per-rank counts: {cpu_hist.tolist()}")

        assert total_output == batch_size
        assert torch.all(new_indices >= 0).item()
        assert rank_counts.tolist() == cpu_hist.tolist()


# World sizes for the edge key checks. The bucketize runs on one GPU for any
# number of buckets. 2**32 and 2**64 leave different remainders modulo 7 and
# 100, so there a 32-bit and a 64-bit reading of a negative key give different
# roundrobin ranks; under hash_roundrobin, at every world size from 2.
EDGE_KEY_WORLD_SIZES = [1, 2, 3, 5, 6, 7, 8, 12, 16, 100, 255]


def edge_keys(key_dtype: torch.dtype) -> torch.Tensor:
    """Keys of *key_dtype* where a 32-bit and a 64-bit reading can disagree.

    The keys next to 0, where a negative key wraps; next to 2**31, where int32
    ends; next to 2**32, where uint32 ends; and next to 2**63, where int64 ends.
    Then random keys over the whole range of the dtype.
    """
    info = np.iinfo(np.int32 if key_dtype == torch.int32 else np.int64)
    edges = set()
    for bound in (0, 1 << 31, 1 << 32, 1 << 63):
        for offset in range(-3, 4):
            edges.update((bound + offset, -(bound + offset)))
    in_range = sorted(key for key in edges if info.min <= key <= info.max)
    random_keys = np.random.default_rng(0).integers(
        info.min, info.max, size=4096, dtype=info.dtype, endpoint=True
    )
    keys = np.concatenate([np.array(in_range, dtype=info.dtype), random_keys])
    return torch.from_numpy(keys).cuda()


def bucketize_one_key_per_sample(
    keys: torch.Tensor, my_size: int, dist_type: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the bucketize kernels with each key in a sample of its own.

    Then the lengths that kernel1 counts say where each key goes: one 1 per
    sample, in the row of its rank. Kernel2 writes the keys rank by rank, and
    the permute it writes says where each key landed.

    Returns the rank of each key, the keys as written, and the permute.
    """
    from dynamicemb.dynamicemb_config import DIST_TYPE_CODES
    from dynamicemb_extensions import block_bucketize_sparse_features

    num_keys = keys.numel()
    (
        new_lengths,
        new_indices,
        _,
        _,
        unbucketize_permute,
    ) = block_bucketize_sparse_features(
        torch.ones(num_keys, dtype=torch.int64, device=keys.device),
        keys,
        bucketize_pos=False,
        sequence=True,
        dist_type_per_feature=torch.tensor(
            [DIST_TYPE_CODES[dist_type]], dtype=torch.int32, device=keys.device
        ),
        block_sizes=torch.tensor(
            [HASH_SIZE // my_size], dtype=keys.dtype, device=keys.device
        ),
        my_size=my_size,
        max_B=1,
    )
    per_rank = new_lengths.view(my_size, num_keys)
    assert torch.all(per_rank.sum(dim=0) == 1).item()
    return per_rank.argmax(dim=0), new_indices, unbucketize_permute


@pytest.mark.parametrize("dist_type", ["roundrobin", "hash_roundrobin"])
@pytest.mark.parametrize("my_size", EDGE_KEY_WORLD_SIZES)
@pytest.mark.parametrize("key_dtype", [torch.int64, torch.int32], ids=str)
def test_bucketize_kernel_matches_key_ownership(key_dtype, my_size, dist_type):
    """The bucketize gives each key the rank that key_ownership gives it.

    key_ownership places the keys that a checkpoint or a delta holds. Those are
    the keys as a table stores them: widened to int64 with their sign. The
    bucketize places the input key in the KJT's own dtype. For an int32 key the
    two have to agree as well, or a loaded key sits on one rank and is looked up
    on another.
    """
    pytest.importorskip("dynamicemb_extensions")
    from dynamicemb.key_ownership import owning_shard

    keys = edge_keys(key_dtype)
    ranks, new_indices, unbucketize_permute = bucketize_one_key_per_sample(
        keys, my_size, dist_type
    )

    stored_keys = keys.to(torch.int64)
    expected = owning_shard(stored_keys, my_size, dist_type)
    mismatch = torch.nonzero(ranks != expected).flatten().tolist()
    assert not mismatch, (
        f"{len(mismatch)} of {keys.numel()} keys get another rank from the "
        f"bucketize than from key_ownership. First: key "
        f"{keys[mismatch[0]].item()}, bucketize rank {ranks[mismatch[0]].item()}, "
        f"key_ownership rank {expected[mismatch[0]].item()}"
    )
    # The reference in this file is written apart from key_ownership, so the
    # kernel and key_ownership cannot be wrong in the same way and still pass.
    reference = assign_owner_cpu(
        stored_keys.cpu().numpy(), my_size, dist_type, HASH_SIZE // my_size
    )
    assert ranks.cpu().numpy().tolist() == reference.tolist()

    # Kernel2 puts each key in the rank kernel1 counted it for, unchanged and in
    # its own dtype.
    assert new_indices.dtype == key_dtype
    order = torch.sort(expected, stable=True).indices
    assert torch.equal(new_indices, keys[order])
    assert torch.equal(new_indices[unbucketize_permute.long()], keys)


@pytest.mark.parametrize("my_size", EDGE_KEY_WORLD_SIZES)
@pytest.mark.parametrize("key_dtype", [torch.int64, torch.int32], ids=str)
def test_bucketize_kernel_continuous_keeps_key_width(key_dtype, my_size):
    """``continuous`` reads a key in its own width and rewrites it in that width.

    key_ownership has no rule for it: it rewrites a key to its offset inside a
    block, so what a table stores is not the input key. The rewrite of an int32
    key is an int32.
    """
    pytest.importorskip("dynamicemb_extensions")

    keys = edge_keys(key_dtype)
    ranks, new_indices, unbucketize_permute = bucketize_one_key_per_sample(
        keys, my_size, "continuous"
    )

    expected_ranks, rewritten = continuous_owner_cpu(
        keys.cpu().numpy(), key_dtype, my_size, HASH_SIZE // my_size
    )
    assert ranks.cpu().numpy().tolist() == expected_ranks.tolist()
    assert new_indices.dtype == key_dtype
    order = np.argsort(expected_ranks, kind="stable")
    assert new_indices.cpu().numpy().tolist() == rewritten[order].tolist()
    permute = unbucketize_permute.long().cpu().numpy()
    assert new_indices.cpu().numpy()[permute].tolist() == rewritten.tolist()
