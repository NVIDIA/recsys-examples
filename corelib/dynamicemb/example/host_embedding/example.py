# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone CPU/CUDA HostLayer export and replay, with no TorchRec dependency."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from dynamicemb.exportable_tables import (
    EmbeddingCollectionIndexerType,
    InferenceEmbeddingCollectionConfig,
    create_inference_embedding_collection,
)
from dynamicemb.exportable_embedding import (
    export_embedding_collection_aot,
    load_embedding_collection_aot,
)


def replay(package, device):
    # All three objects own resources referenced by the compiled model.
    loader, layers, indexers = load_embedding_collection_aot(package, device)
    for case in json.loads((package / "replay_cases.json").read_text()):
        inputs = [
            torch.tensor(value, dtype=torch.int64, device=device)
            for value in case["inputs"]
        ]
        expected = torch.tensor(case["expected"], dtype=torch.float32)
        result = loader.run(inputs)[0].cpu().float()
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
    print(f"Verified Python HostLayer AOT replay on {device}")


def export(args):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    count = 1 if args.indexer == "identity" else 2
    tables = [
        SimpleNamespace(
            name=f"table_{i}",
            num_embeddings=4,
            embedding_dim=8,
            feature_names=[f"feature_{i}"],
        )
        for i in range(count)
    ]
    model = create_inference_embedding_collection(
        tables,
        config=InferenceEmbeddingCollectionConfig(
            indexer_type=EmbeddingCollectionIndexerType(args.indexer),
            nve_layer_type="host",
            indexer_state_sidecar=not args.embedded,
        ),
        pooling_mode=args.pooling,
        output_dtype=dtype,
        device=device,
    ).eval()
    weights = (torch.arange(count * 4 * 8).reshape(-1, 8) / 8).to(dtype)
    model.load_from_embedding_table(weights)
    cases = []
    examples = []
    for local_keys in ([0, 3, 2, 1], [3, 0, 9, -8, 1, 2], [1, 0]):
        keys = torch.tensor(local_keys * count, device=device)
        offsets = torch.arange(count + 1, device=device) * len(local_keys)
        inputs = [keys, offsets]
        expected = torch.stack(
            [
                (
                    weights[table * 4 + key]
                    if 0 <= key < 4
                    else torch.zeros(8, dtype=dtype)
                )
                for table in range(count)
                for key in local_keys
            ]
        )
        if args.pooling != -1:
            inputs.append(torch.arange(0, keys.numel() + 1, 2, device=device))
            expected = expected.reshape(-1, 2, 8).sum(1)
            if args.pooling == 2:
                expected /= 2
        torch.testing.assert_close(model(*inputs).cpu(), expected)
        cases.append(
            {
                "inputs": [value.cpu().tolist() for value in inputs],
                "expected": expected.float().tolist(),
            }
        )
        examples.append(tuple(inputs))
    n = torch.export.Dim("pairs", min=1, max=32)
    shapes = ({0: 2 * n}, None)
    if args.pooling != -1:
        shapes += ({0: n + 1},)
    args.package_dir.mkdir(parents=True, exist_ok=True)
    indexers = export_embedding_collection_aot(
        model,
        examples[0],
        args.package_dir,
        dynamic_shapes=shapes,
    )
    (args.package_dir / "replay_cases.json").write_text(json.dumps(cases))
    print(f"Exported {args.indexer} HostLayer on {device} to {args.package_dir}")
    replay(args.package_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument(
        "--indexer",
        choices=["fused_identity", "linear_hash_map", "identity"],
        default="linear_hash_map",
    )
    parser.add_argument(
        "--embedded",
        action="store_true",
        help="Keep index state in the model instead of a sidecar",
    )
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--pooling", type=int, choices=[-1, 1, 2], default=-1)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay an existing package in a fresh process",
    )
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    if device.type == "cuda":
        # Existing NVE/indexer kernels use the calling thread's current device.
        torch.cuda.set_device(device)
    if arguments.replay:
        replay(arguments.package_dir, torch.device(arguments.device))
    else:
        export(arguments)
