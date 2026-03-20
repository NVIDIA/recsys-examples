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
- one AOTI artifact per pooling mode (-1/1/2); separate demos provided for non-pooled and sum-pooling
"""

import os
import shutil
import tempfile
from typing import List, Optional

import torch
import torch.distributed as dist

from inference_embedding_impl import (
    BatchedDynamicEmbeddingTablesV2,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    InferenceEmbeddingTable,
    ScorePolicy,
    _load_inference_emb_ops,
    _load_nve_torch_bindings,
    encode_checkpoint_file_path,
    encode_meta_json_file_path,
)


def _ensure_single_process_group() -> Optional[str]:
    if dist.is_initialized():
        return None

    init_dir = tempfile.mkdtemp(prefix="dynamicemb_pg_")
    init_method = f"file://{os.path.join(init_dir, 'init')}"
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=0,
        world_size=1,
    )
    return init_dir


def _cleanup_single_process_group(init_dir: Optional[str]) -> None:
    if init_dir is None:
        return
    if dist.is_initialized():
        dist.destroy_process_group()
    try:
        os.rmdir(init_dir)
    except OSError:
        pass


def _build_e2e_batched_tables(
    table_names: List[str],
    capacity_list: List[int],
    embedding_dim: int,
    device: torch.device,
) -> "BatchedDynamicEmbeddingTablesV2":
    local_hbm_for_values = 1024**3
    table_options = [
        DynamicEmbTableOptions(
            training=False,
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            dim=embedding_dim,
            init_capacity=capacity,
            max_capacity=capacity,
            bucket_capacity=128,
            score_strategy=DynamicEmbScoreStrategy.STEP,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.CONSTANT,
                value=0.0,
            ),
            local_hbm_for_values=local_hbm_for_values,
            device_id=device.index,
        )
        for capacity in capacity_list
    ]

    return BatchedDynamicEmbeddingTablesV2(
        table_options=table_options,
        table_names=table_names,
        output_dtype=torch.float32,
        device=device,
    )


def _make_expected_payload(
    table_id: int,
    num_embeddings: int,
    embedding_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_key = (table_id + 1) * 100000
    keys = torch.arange(base_key, base_key + num_embeddings, device=device).to(torch.int64)

    feature_axis = torch.arange(embedding_dim, device=device, dtype=torch.float32)
    embeddings = (
        keys.to(torch.float32).unsqueeze(1) * 0.001
        + feature_axis.unsqueeze(0) * 0.01
        + float(table_id)
    ).contiguous()
    scores = torch.arange(
        1000 + table_id * 100,
        1000 + table_id * 100 + num_embeddings,
    ).to(dtype=torch.uint64, device=device)
    return keys, embeddings, scores


def _unwrap_single_output(output):
    if isinstance(output, (tuple, list)):
        if len(output) != 1:
            raise RuntimeError(
                f"Expected a single output from the packaged model, got {len(output)}"
            )
        return output[0]
    return output


def _save_tensor_cpp_compatible(tensor: torch.Tensor, path: str) -> None:
    """Save a tensor in TorchScript zip format loadable by torch::jit::load() in C++.

    ``torch.save(tensor, path)`` uses Python pickle format which is NOT compatible
    with C++'s ``torch::load(tensor, path)`` / ``torch::serialize::InputArchive``.
    Wrapping the tensor as a registered buffer in a scripted nn.Module and calling
    ``torch.jit.save`` produces the TorchScript zip archive that C++ expects.
    """

    class _TensorHolder(torch.nn.Module):
        def __init__(self, t: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("tensor", t)

        def forward(self) -> torch.Tensor:
            return self.tensor  # type: ignore[return-value]

    holder = torch.jit.script(_TensorHolder(tensor.cpu().contiguous()))
    torch.jit.save(holder, path)


def test_inference_emb():
    if not _load_inference_emb_ops():
        raise RuntimeError("inference_emb_ops.so must be loaded before running the E2E load test")
    if not _load_nve_torch_bindings():
        raise RuntimeError("libnve_torch.so must be loaded before running the E2E load test")
    if "BatchedDynamicEmbeddingTablesV2" not in globals():
        raise RuntimeError("dynamicemb imports are unavailable for the E2E load test")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the E2E load test")

    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    embedding_dim = 128
    dump_root_dir = os.path.join(
        os.path.dirname(__file__),
        "inference_emb_dump",
    )

    if os.path.exists(dump_root_dir):
        if os.path.isdir(dump_root_dir):
            shutil.rmtree(dump_root_dir)
        else:
            os.remove(dump_root_dir)
    os.makedirs(dump_root_dir, exist_ok=True)

    for num_tables, counts in ((2, [200, 260]), (3, [120, 240, 360])):
        table_names = [f"table_{i}" for i in range(num_tables)]
        capacity_list = [512 for _ in range(num_tables)]
        save_dir = os.path.join(dump_root_dir, f"num_tables_{num_tables}")
        os.makedirs(save_dir, exist_ok=True)

        batched_tables = _build_e2e_batched_tables(
            table_names,
            capacity_list,
            embedding_dim,
            device,
        )

        expected_payloads = []
        for table_id, num_embeddings in enumerate(counts):
            keys, embeddings, scores = _make_expected_payload(
                table_id,
                num_embeddings,
                embedding_dim,
                device,
            )
            table_ids = torch.full(
                (num_embeddings,),
                table_id,
                dtype=torch.int64,
                device=device,
            )
            batched_tables.tables.insert(keys, table_ids, embeddings, scores)
            expected_payloads.append((keys, embeddings, scores))

        feature_table_map = list(range(num_tables))

        pg_init_dir = _ensure_single_process_group()
        try:
            batched_tables.dump(
                save_dir,
                optim=False,
                counter=False,
                table_names=table_names,
            )

            for table_name in table_names:
                meta_path = encode_meta_json_file_path(save_dir, table_name)
                key_path = encode_checkpoint_file_path(
                    save_dir, table_name, 0, 1, "keys"
                )
                value_path = encode_checkpoint_file_path(
                    save_dir, table_name, 0, 1, "values"
                )
                score_path = encode_checkpoint_file_path(
                    save_dir, table_name, 0, 1, "scores"
                )
                assert os.path.exists(meta_path), f"Missing meta file: {meta_path}"
                assert os.path.exists(key_path), f"Missing key file: {key_path}"
                assert os.path.exists(value_path), f"Missing value file: {value_path}"
                assert os.path.exists(score_path), f"Missing score file: {score_path}"
                assert os.path.getsize(key_path) > 0, f"Empty key file: {key_path}"
                assert os.path.getsize(value_path) > 0, f"Empty value file: {value_path}"
                assert os.path.getsize(score_path) > 0, f"Empty score file: {score_path}"

            inference_table = InferenceEmbeddingTable(
                table_options=[
                    DynamicEmbTableOptions(
                        init_capacity=cap,
                        max_capacity=cap,
                        dim=embedding_dim,
                        embedding_dtype=torch.float32,
                        global_hbm_for_values=1 << 28,
                    )
                    for cap in capacity_list
                ],
                pooling_mode=-1,
                table_names=table_names,
                feature_table_map=feature_table_map,
                device=device,
            )
            inference_table.load(save_dir, table_names=table_names)
            print("✓ Checkpoint load completed without error")

            lookup_keys = torch.cat(
                [payload[0] for payload in expected_payloads],
                dim=0,
            )
            expected_lookup_embeddings = torch.cat(
                [payload[1] for payload in expected_payloads],
                dim=0,
            )
            lookup_offsets_list = [0]
            for keys, _embeddings, _scores in expected_payloads:
                lookup_offsets_list.append(lookup_offsets_list[-1] + keys.numel())
            lookup_offsets = torch.tensor(
                lookup_offsets_list,
                dtype=torch.int64,
                device=device,
            )

            eager_e2e_output = inference_table(lookup_keys, lookup_offsets)
            torch.testing.assert_close(eager_e2e_output, expected_lookup_embeddings)

            keys_tensor_path = os.path.join(save_dir, "keys.pt")
            offsets_tensor_path = os.path.join(save_dir, "offsets.pt")
            embeddings_tensor_path = os.path.join(save_dir, "embeddings.pt")
            _save_tensor_cpp_compatible(lookup_keys, keys_tensor_path)
            _save_tensor_cpp_compatible(lookup_offsets, offsets_tensor_path)
            _save_tensor_cpp_compatible(expected_lookup_embeddings, embeddings_tensor_path)

            aoti_model_path = os.path.join(save_dir, "model.pt2")
            if os.path.exists(aoti_model_path):
                os.remove(aoti_model_path)

            exported = torch.export.export(
                inference_table,
                (lookup_keys, lookup_offsets),
                dynamic_shapes={
                    "keys": {0: torch.export.dynamic_shapes.Dim.AUTO,},
                    "offsets": {0: torch.export.dynamic_shapes.Dim.AUTO,},
                },
            )
            aoti_output_path = torch._inductor.aoti_compile_and_package(
                exported,
                package_path=aoti_model_path,
            )
            compiled_model = torch._inductor.aoti_load_package(aoti_output_path)
            compiled_output = _unwrap_single_output(
                compiled_model(lookup_keys, lookup_offsets)
            )
            torch.testing.assert_close(compiled_output, expected_lookup_embeddings)
            print(
                "✓ AOTInductor package generated and validated for "
                f"{table_names}: {aoti_output_path}"
            )
            print(
                "✓ Saved C++ comparison tensors: "
                f"{keys_tensor_path}, {offsets_tensor_path}, {embeddings_tensor_path}"
            )

            for table_id, table_name in enumerate(table_names):
                keys, expected_embeddings, expected_scores = expected_payloads[table_id]
                table_ids = torch.full(
                    (keys.numel(),),
                    table_id,
                    dtype=torch.int64,
                    device=device,
                )

                loaded_scores, founds, indices = inference_table.hash_table.lookup(
                    keys=keys,
                    table_ids=table_ids,
                    score_value=None,
                    score_policy=int(ScorePolicy.CONST),
                )
                assert torch.all(founds), f"Missing loaded keys for {table_name}"
                indices += inference_table.table_offsets_[table_id]  # Adjust indices by table offset

                loaded_embeddings = inference_table.nve_embedding_.lookup(indices.to(torch.int64))
                torch.testing.assert_close(loaded_embeddings, expected_embeddings)
                torch.testing.assert_close(
                    loaded_scores.to(torch.uint64),
                    expected_scores,
                )

                negative_keys = keys + 9000000
                _, negative_founds, _ = inference_table.hash_table.lookup(
                    keys=negative_keys,
                    table_ids=table_ids,
                    score_value=None,
                    score_policy=int(ScorePolicy.CONST),
                )
                assert not torch.any(
                    negative_founds
                ), f"Unexpected lookup hit for unseen keys in {table_name}"

                negative_offsets = torch.tensor(
                    [0, negative_keys.numel()],
                    dtype=torch.int64,
                    device=device,
                )
                negative_output = inference_table(negative_keys, negative_offsets)
                torch.testing.assert_close(negative_output, torch.zeros_like(negative_output))
        finally:
            _cleanup_single_process_group(pg_init_dir)

    print(f"✓ Kept dumped checkpoint files under: {dump_root_dir}")
    

def test_inference_emb_pooled_sum():
    """Test sum pooling export and AOTI compilation.

    Builds an ``InferenceEmbeddingTable`` with ``pooling_mode=1`` (sum), groups
    loaded keys into 2-key pooling bags, and verifies:

    1. Eager forward output matches the expected element-wise sum of embedding pairs.
    2. ``torch.export`` succeeds and records a graph containing ``lookup_with_pooling``.
    3. The AOTI-compiled artifact produces output identical to the eager run.

    Saved artifacts per ``num_tables`` variant (under ``inference_emb_dump_pooled_sum/num_tables_N/``):

    - ``model_pooled_sum.pt2``   – AOTI packaged model
    - ``keys.pt``                – flat lookup keys (C++ comparison input)
    - ``offsets.pt``             – per-table CSR boundaries (C++ comparison input)
    - ``pooling_offsets.pt``     – per-bag CSR boundaries (C++ comparison input)
    - ``pooled_embeddings.pt``   – expected sum-pooled output (C++ comparison reference)
    """
    if not _load_inference_emb_ops():
        raise RuntimeError(
            "inference_emb_ops.so must be loaded before running the pooled export test"
        )
    if not _load_nve_torch_bindings():
        raise RuntimeError(
            "libnve_torch.so must be loaded before running the pooled export test"
        )
    if "BatchedDynamicEmbeddingTablesV2" not in globals():
        raise RuntimeError(
            "dynamicemb imports are unavailable for the pooled export test"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the pooled export test")

    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    embedding_dim = 128
    dump_root_dir = os.path.join(
        os.path.dirname(__file__),
        "inference_emb_dump_pooled_sum",
    )

    if os.path.exists(dump_root_dir):
        if os.path.isdir(dump_root_dir):
            shutil.rmtree(dump_root_dir)
        else:
            os.remove(dump_root_dir)
    os.makedirs(dump_root_dir, exist_ok=True)

    # Total key counts per case are even so that 2-key bags divide them exactly.
    for num_tables, counts in ((2, [200, 260]), (3, [120, 240, 360])):
        table_names = [f"table_{i}" for i in range(num_tables)]
        capacity_list = [512 for _ in range(num_tables)]
        save_dir = os.path.join(dump_root_dir, f"num_tables_{num_tables}")
        os.makedirs(save_dir, exist_ok=True)

        batched_tables = _build_e2e_batched_tables(
            table_names,
            capacity_list,
            embedding_dim,
            device,
        )

        expected_payloads = []
        for table_id, num_embeddings in enumerate(counts):
            keys, embeddings, scores = _make_expected_payload(
                table_id,
                num_embeddings,
                embedding_dim,
                device,
            )
            table_ids = torch.full(
                (num_embeddings,),
                table_id,
                dtype=torch.int64,
                device=device,
            )
            batched_tables.tables.insert(keys, table_ids, embeddings, scores)
            expected_payloads.append((keys, embeddings, scores))

        feature_table_map = list(range(num_tables))

        pg_init_dir = _ensure_single_process_group()
        try:
            batched_tables.dump(
                save_dir,
                optim=False,
                counter=False,
                table_names=table_names,
            )

            inference_table = InferenceEmbeddingTable(
                table_options=[
                    DynamicEmbTableOptions(
                        init_capacity=cap,
                        max_capacity=cap,
                        dim=embedding_dim,
                        embedding_dtype=torch.float32,
                        global_hbm_for_values=1 << 28,
                    )
                    for cap in capacity_list
                ],
                pooling_mode=1,
                table_names=table_names,
                feature_table_map=feature_table_map,
                device=device,
            )
            inference_table.load(save_dir, table_names=table_names)
            print("✓ Checkpoint load completed without error (pooled sum test)")

            # Build flat key / per-table offset tensors.
            lookup_keys = torch.cat(
                [payload[0] for payload in expected_payloads],
                dim=0,
            )
            lookup_offsets_list = [0]
            for keys, _embeddings, _scores in expected_payloads:
                lookup_offsets_list.append(lookup_offsets_list[-1] + keys.numel())
            lookup_offsets = torch.tensor(
                lookup_offsets_list,
                dtype=torch.int64,
                device=device,
            )

            total_keys = lookup_keys.numel()
            assert total_keys % 2 == 0, (
                f"Expected even total key count for 2-key bags, got {total_keys}"
            )

            # 2-key pooling bags: bag i covers global_indices[2*i : 2*i+2].
            lookup_pooling_offsets = torch.arange(
                0, total_keys + 1, 2, dtype=torch.int64, device=device
            )

            # Expected output: element-wise sum of consecutive embedding pairs.
            expected_individual = torch.cat(
                [payload[1] for payload in expected_payloads], dim=0
            )
            expected_pooled = expected_individual[0::2] + expected_individual[1::2]

            # Eager validation.
            eager_output = inference_table(
                lookup_keys, lookup_offsets, lookup_pooling_offsets
            )
            torch.testing.assert_close(eager_output, expected_pooled)
            print("✓ Eager sum-pooled forward matches expected output")

            # Save C++-readable comparison tensors.
            keys_tensor_path = os.path.join(save_dir, "keys.pt")
            offsets_tensor_path = os.path.join(save_dir, "offsets.pt")
            pooling_offsets_tensor_path = os.path.join(save_dir, "pooling_offsets.pt")
            pooled_embeddings_tensor_path = os.path.join(save_dir, "pooled_embeddings.pt")
            _save_tensor_cpp_compatible(lookup_keys, keys_tensor_path)
            _save_tensor_cpp_compatible(lookup_offsets, offsets_tensor_path)
            _save_tensor_cpp_compatible(lookup_pooling_offsets, pooling_offsets_tensor_path)
            _save_tensor_cpp_compatible(expected_pooled, pooled_embeddings_tensor_path)

            aoti_model_path = os.path.join(save_dir, "model_pooled_sum.pt2")
            if os.path.exists(aoti_model_path):
                os.remove(aoti_model_path)

            exported = torch.export.export(
                inference_table,
                (lookup_keys, lookup_offsets, lookup_pooling_offsets),
                dynamic_shapes={
                    "keys": {0: torch.export.dynamic_shapes.Dim.AUTO},
                    "offsets": {0: torch.export.dynamic_shapes.Dim.AUTO},
                    "pooling_offsets": {0: torch.export.dynamic_shapes.Dim.AUTO},
                },
            )
            aoti_output_path = torch._inductor.aoti_compile_and_package(
                exported,
                package_path=aoti_model_path,
            )
            compiled_model = torch._inductor.aoti_load_package(aoti_output_path)
            compiled_output = _unwrap_single_output(
                compiled_model(lookup_keys, lookup_offsets, lookup_pooling_offsets)
            )
            torch.testing.assert_close(compiled_output, expected_pooled)
            print(
                "✓ AOTInductor sum-pooling package generated and validated for "
                f"{table_names}: {aoti_output_path}"
            )
            print(
                "✓ Saved C++ comparison tensors: "
                f"{keys_tensor_path}, {offsets_tensor_path}, "
                f"{pooling_offsets_tensor_path}, {pooled_embeddings_tensor_path}"
            )
        finally:
            _cleanup_single_process_group(pg_init_dir)

    print(f"✓ Kept dumped checkpoint files under: {dump_root_dir}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Test Export Demo: INFERENCE_EMB Custom Operators")
    print("=" * 60)

    test_inference_emb()
    test_inference_emb_pooled_sum()

    print("\n" + "=" * 60)
    print("Tests completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
