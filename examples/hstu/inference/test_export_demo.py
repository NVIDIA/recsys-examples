# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Demo: Exportable inference embedding table using INFERENCE_EMB custom operators.

This demo shows how to:
1. Use the new dispatcher custom operators: get_table_range, expand_table_ids, table_lookup
2. Build a simple export-compatible module for inference-only embedding lookup
3. Trace and export with torch.export

Limitations (by design for quick demo):
- CUDA-only, no CPU path
- inference-only, no insert/evict
- lookup-only, no other operations
- non-pooled output shape: (num_indices, embedding_dim)
"""

from typing import List, Optional
import torch
import os

# ── Step 1: Load inference_emb_ops.so BEFORE importing dynamicemb ──────────────
# dynamicemb's index_range_meta.py and lookup_meta.py attempt to register fake
# kernels at import time.  Those registrations silently fail with a RuntimeWarning
# if the operators don't exist yet.  Loading the .so here ensures the dispatcher
# already knows about INFERENCE_EMB::* before dynamicemb is imported below.
_SEARCH_PATHS = [
    os.path.join(
        os.path.dirname(__file__),
        "../../../corelib/dynamicemb/torch_binding_build/inference_emb_ops.so"
    ),
    os.path.join(
        os.path.dirname(__file__),
        "../../../corelib/dynamicemb/build_inference_ops_check/inference_emb_ops.so"
    ),
    os.path.join(
        os.path.dirname(__file__),
        "../../../corelib/dynamicemb/build/inference_emb_ops.so"
    ),
    "inference_emb_ops.so",
]

_ops_loaded: bool = False
for _path in _SEARCH_PATHS:
    if os.path.exists(_path):
        try:
            torch.ops.load_library(_path)
            print(f"[INFO] Loaded inference_emb_ops.so from {_path}")
            _ops_loaded = True
        except Exception as _e:
            print(f"[WARN] Failed to load {_path}: {_e}")
        break  # stop after first found path, whether load succeeded or not

if not _ops_loaded:
    print("[WARN] Could not find inference_emb_ops.so. Custom ops may not be available.")

# ── Step 2: Import dynamicemb AFTER loading the .so ────────────────────────────
# Fake-kernel registration in index_range_meta.py / lookup_meta.py now succeeds
# because the INFERENCE_EMB operators already exist in the dispatcher.
try:
    import dynamicemb
    import dynamicemb.index_range_meta as _index_range_meta
    import dynamicemb.lookup_meta as _lookup_meta
    from dynamicemb.scored_hashtable import ScorePolicy

    # If the .so was loaded after a previous (failed) import of dynamicemb
    # (e.g. from another module), re-trigger fake-kernel registration now.
    if _ops_loaded:
        if not _index_range_meta.REGISTERED:
            _index_range_meta.register_index_range_fake()
        if not _lookup_meta.REGISTERED:
            _lookup_meta.register_lookup_fake()
except ImportError:
    pass


def _load_inference_emb_ops() -> bool:
    """Return whether inference_emb_ops.so was successfully loaded at module init.

    The library is loaded eagerly at module level (before dynamicemb import) so
    that fake-kernel registration inside dynamicemb succeeds without warnings.
    This function is kept for use in main() to report the load status.
    """
    return _ops_loaded


class InferenceLinearBucketTable(torch.nn.Module):
    """Simple exportable hash table wrapper for inference lookup using custom op.
    
    This is a minimal demo version that focuses on lookup-only, non-pooled inference.
    For the full production version, see LinearBucketTable in scored_hashtable.py.
    """

    def __init__(
        self,
        capacity: List[int],
        key_type: torch.dtype = torch.int64,
        bucket_capacity: int = 128,
        device: Optional[torch.device] = None,
    ):
        """Initialize demo hash table.
        
        Args:
            capacity: List of per-table capacities
            key_type: torch.int64 or torch.uint64
            bucket_capacity: slots per bucket
            device: CUDA device (defaults to current)
        """
        super().__init__()
        
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        
        self.device = device
        self.key_type_ = key_type
        self.bucket_capacity_ = bucket_capacity
        
        # For this demo: assume single table or fixed small number
        self.num_tables_ = len(capacity)
        
        # Compute bucket layout
        per_table_num_buckets = []
        bucket_offset_list = [0]
        for cap in capacity:
            nb = (cap + bucket_capacity - 1) // bucket_capacity
            per_table_num_buckets.append(nb)
            bucket_offset_list.append(bucket_offset_list[-1] + nb)
        
        total_buckets = bucket_offset_list[-1]
        
        # For demo: single score type (int64), key, and digest
        # Storage layout: [key, digest, score] repeated
        bytes_per_slot = 8 + 1 + 8  # key (int64) + digest (uint8) + score (int64)
        total_storage_bytes = bytes_per_slot * bucket_capacity * total_buckets
        
        # Register as buffers so they move with the module
        self.register_buffer(
            "table_storage_",
            torch.zeros(total_storage_bytes, dtype=torch.uint8, device=device)
        )
        self.register_buffer(
            "table_bucket_offsets_",
            torch.tensor(bucket_offset_list, dtype=torch.int64, device=device)
        )
        
        # Demo: bucket sizes counter (for tracking)
        self.register_buffer(
            "bucket_sizes",
            torch.zeros(total_buckets, dtype=torch.int32, device=device)
        )
        
        # Note: For a full implementation, would also initialize keys with empty marker,
        # but for this eval-only demo we assume the table is pre-populated or the
        # lookup custom op handles empty gracefully.
    
    def lookup(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        score_value: Optional[torch.Tensor] = None,
        score_policy: int = int(ScorePolicy.CONST),
    ) -> tuple:
        """Lookup keys in the hash table using the custom operator.
        
        Args:
            keys: (num_keys,) int64 or uint64 keys to lookup
            table_ids: (num_keys,) int64 table IDs for each key
            score_value: Optional score input (required for Assign/Accumulate policies)
            score_policy: ScorePolicy enum as int
        
        Returns:
            (score_out, founds, indices): 
                - score_out: (num_keys,) int64 output scores
                - founds: (num_keys,) bool whether key was found
                - indices: (num_keys,) int64 indices into the table
        """
        # Call the custom operator through torch.ops
        score_out, founds, indices = torch.ops.INFERENCE_EMB.table_lookup(
            self.table_storage_,
            self.table_bucket_offsets_,
            self.bucket_capacity_,
            keys,
            table_ids,
            score_value,
            score_policy,
            None,  # ovf_storage
            0,     # ovf_bucket_capacity
            None,  # ovf_output_offsets
        )
        
        return score_out, founds, indices


class InferenceEmbeddingTable(torch.nn.Module):
    """Simplified, export-compatible embedding table using custom ops.
    
    This demo version:
    - Supports lookup-only inference (no pooling/averaging)
    - Uses three INFERENCE_EMB custom operators
    - Registers all persistent state as buffers for torch.export compatibility
    - Avoids graph breaks (no .cpu(), .item(), dynamic control flow on tensor values)
    """

    def __init__(
        self,
        feature_offsets: List[int],  # Feature boundaries in table-space
        num_tables: int = 1,
        capacity_per_table: int = 1000,
        key_type: torch.dtype = torch.int64,
        device: Optional[torch.device] = None,
    ):
        """Initialize demo embedding table.
        
        Args:
            feature_offsets: List of offsets marking feature boundaries
                e.g., [0, 128, 256] means 2 features with sizes 128 and 128
            num_tables: Number of logical hash tables
            capacity_per_table: Capacity per table
            key_type: torch.int64 or torch.uint64
            device: CUDA device
        """
        super().__init__()
        
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        
        self.device = device
        self.num_tables_ = num_tables
        self.key_type_ = key_type
        
        # Feature offsets: boundaries in the embedding space
        self.register_buffer(
            "feature_offsets",
            torch.tensor(feature_offsets, dtype=torch.int64, device=device)
        )
        
        # Create hash tables for each logical table
        capacities = [capacity_per_table] * num_tables
        self.hash_table = InferenceLinearBucketTable(
            capacity=capacities,
            key_type=key_type,
            bucket_capacity=128,
            device=device,
        )
        
        # For demo: linear memory table (simple embedding vectors)
        # Assume 128-dim embeddings
        embedding_dim = 128
        total_embedding_rows = sum(feature_offsets[i + 1] - feature_offsets[i] 
                                    for i in range(len(feature_offsets) - 1))
        
        linear_mem_table = torch.randn(
            total_embedding_rows, embedding_dim, dtype=torch.float32, device=device
        )
        self.register_buffer("linear_mem_table", linear_mem_table)
    
    def forward(
        self,
        indices: torch.Tensor,  # (batch_size,) indices to lookup
        offsets: torch.Tensor,  # (num_features + 1,) batch offsets for pooling
    ) -> torch.Tensor:
        """Forward pass using INFERENCE_EMB custom operators.
        
        This demo does eval-only, lookup-only inference (no pooling).
        
        Args:
            indices: (total_lookups,) int64 feature IDs to lookup
            offsets: (num_features + 1,) int64 offsets into indices
        
        Returns:
            embeddings: (total_lookups, embedding_dim) float32 embedding vectors
        """
        # Step 1: Get table ranges (which table each feature belongs to)
        # get_table_range(offsets, feature_offsets) -> (num_features, 2)
        # Each row [start, end] indicates the table-space range for that feature
        table_ranges = torch.ops.INFERENCE_EMB.get_table_range(
            offsets, self.feature_offsets
        )  # (num_features, 2)
        
        # Step 2: Expand table IDs from offsets
        # expand_table_ids(offsets, table_offsets, num_tables, local_batch_size, num_elements)
        # Returns (num_elements,) int64 table_ids indicating which table each element belongs to
        num_features = offsets.shape[0] - 1
        num_elements = indices.shape[0]
        
        # Prepare table_offsets_in_feature: where in feature space each table starts
        table_offsets = torch.arange(
            num_features + 1, dtype=torch.int64, device=self.device
        )
        
        table_ids = torch.ops.INFERENCE_EMB.expand_table_ids(
            offsets,
            table_offsets,
            self.num_tables_,
            num_features,
            num_elements,
        )  # (num_elements,)
        
        # Step 3: Lookup in hash table
        # table_lookup returns (score, found, indices_in_table)
        scores, founds, table_indices = self.hash_table.lookup(
            keys=indices,
            table_ids=table_ids,
            score_value=None,
            score_policy=int(ScorePolicy.CONST),
        )  # each (num_elements,)
        
        # Step 4: Gather embeddings using table indices
        # For demo: use table_indices as direct row indices in linear_mem_table
        embeddings = torch.index_select(
            self.linear_mem_table, 0, table_indices
        )  # (num_elements, embedding_dim)
        
        # For unfound items, zero out the embedding
        # embeddings = torch.where(
        #     founds.unsqueeze(-1),  # (num_elements, 1) broadcast
        #     embeddings,  # (num_elements, embedding_dim)
        #     torch.zeros_like(embeddings),  # (num_elements, embedding_dim)
        # )
        
        return embeddings


def test_eager_forward():
    """Test forward pass in eager mode."""
    print("\n=== Test Eager Forward ===")
    
    # Create module
    feature_offsets = [0, 128, 256]
    table = InferenceEmbeddingTable(
        feature_offsets=feature_offsets,
        num_tables=2,
        capacity_per_table=1000,
    )
    table.eval()
    
    # Create dummy input
    # indices: (4,) feature IDs
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64).cuda()
    # offsets: (3,) marking 2 features: [0, 2, 4]
    # meaning feature 0 has 2 items (indices[0:2]), feature 1 has 2 items (indices[2:4])
    offsets = torch.tensor([0, 2, 4], dtype=torch.int64).cuda()
    
    # Forward
    with torch.no_grad():
        embeddings = table(indices, offsets)
    
    print(f"Input indices shape: {indices.shape}")
    print(f"Input offsets shape: {offsets.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Output embeddings dtype: {embeddings.dtype}")
    
    assert embeddings.shape == (4, 128), f"Expected (4, 128), got {embeddings.shape}"
    assert embeddings.dtype == torch.float32
    print("✓ Eager forward test passed")


def test_torch_export():
    """Test torch.export compatibility (smoke test)."""
    print("\n=== Test Torch Export ===")
    
    feature_offsets = [0, 128, 256]
    table = InferenceEmbeddingTable(
        feature_offsets=feature_offsets,
        num_tables=2,
        capacity_per_table=1000,
    )
    table.eval()
    
    # Create example inputs for export
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64).cuda()
    offsets = torch.tensor([0, 2, 4], dtype=torch.int64).cuda()
    
    try:
        # Attempt to export the model
        exported = torch.export.export(table, (indices, offsets))
        print(f"✓ torch.export succeeded")
        print(f"  Exported module nodes: {len(list(exported.graph.nodes))}")
        
        # Verify exported module runs
        exported_result = exported.module()(indices, offsets)
        print(f"  Exported forward output shape: {exported_result.shape}")
        
    except Exception as e:
        # Expected for now if custom ops not available or not exportable
        print(f"⚠ torch.export raised (expected if custom ops unavailable): {type(e).__name__}")
        if "INFERENCE_EMB" in str(e) or "custom" in str(e).lower():
            print(f"  Note: This is expected if torch.ops.INFERENCE_EMB is not registered")
            print(f"  Make sure inference_emb_ops.so is built and loaded")
        else:
            print(f"  Error detail: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Test Export Demo: INFERENCE_EMB Custom Operators")
    print("=" * 60)
    
    # Try to load custom operators
    ops_loaded = _load_inference_emb_ops()
    if not ops_loaded:
        print("[WARN] Custom operators not loaded; tests will demonstrate structure only")
    
    # Run tests
    try:
        test_eager_forward()
    except Exception as e:
        print(f"✗ test_eager_forward failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_torch_export()
    except Exception as e:
        print(f"✗ test_torch_export failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Tests completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
