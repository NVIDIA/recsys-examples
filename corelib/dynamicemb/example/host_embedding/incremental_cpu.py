# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU HostLayer incremental loading: continuous fixed mappings, paused hash mappings."""
import argparse
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import torch
from dynamicemb.exportable_embedding import (
    EmbeddingCollectionUpdateSubscriber,
    load_embedding_collection_aot,
)
from dynamicemb.exportable_embedding.incremental_update import (
    EmbeddingCollectionUpdateCoordinator,
)
from example import export as export_baseline


def main(args):
    args.device = "cpu"
    export_baseline(args)
    loader, layers, indexers = load_embedding_collection_aot(
        args.package_dir, torch.device("cpu")
    )
    subscriber = EmbeddingCollectionUpdateSubscriber(
        str(args.package_dir),
        indexers,
        {layer._export_module_path: layer.emb_layer for layer in layers},
        -1,
    )
    coordinator = EmbeddingCollectionUpdateCoordinator.open(
        package_dir=args.package_dir,
        shared_update_dir=args.package_dir / "updates",
        device=torch.device("cpu"),
    )
    collection_id = next(iter(indexers.bindings))
    names = indexers.bindings[collection_id].table_names
    paused = args.indexer == "linear_hash_map"
    baseline = json.loads((args.package_dir / "replay_cases.json").read_text())[0]
    inputs = [torch.tensor(v, dtype=torch.int64) for v in baseline["inputs"]]
    ready, stop = threading.Event(), threading.Event()
    errors, progress = [], [0]

    def infer():
        try:
            while not stop.is_set():
                output = loader.run(inputs)[0]
                assert torch.isfinite(
                    output
                ).all()  # Stale/mixed rows are allowed here.
                progress[0] += 1
                ready.set()
        except BaseException as error:
            errors.append(error)
            ready.set()

    worker = None
    if not paused:
        worker = threading.Thread(target=infer)
        worker.start()
        assert ready.wait(30)
    updates = []
    previous = 1
    before = progress[0]
    try:
        for generation in range(1, 33):
            key = 1000 + generation if paused else 1
            delta = SimpleNamespace(
                table_names=names,
                keys=[torch.tensor([0, key])] * len(names),
                values=[
                    torch.full((2, 8), float(generation + i)) for i in range(len(names))
                ],
                evicted_keys=[torch.tensor([previous])] * len(names)
                if paused
                else [None] * len(names),
                erased_keys=[None] * len(names),
            )
            update = coordinator.apply_delta(collection_id, delta)
            subscriber.apply_incremental_load(update.to_json(), inference_paused=paused)
            subscriber.wait_for_retirement(collection_id, update.snapshot_id)
            updates.append(json.loads(update.to_json()))
            previous = key
    finally:
        stop.set()
        if worker:
            worker.join(30)
            assert not worker.is_alive()
    assert not errors, errors
    if not paused:
        assert progress[0] > before, "Inference did not progress during loading"
    local_keys = [0, previous, 1 if paused else 0, previous]
    keys = torch.tensor(local_keys * len(names))
    offsets = torch.arange(len(names) + 1) * 4
    args_in = [keys, offsets]
    expected = torch.tensor(
        [
            [float(32 + i) if key in (0, previous) else 0.0] * 8
            for i in range(len(names))
            for key in local_keys
        ],
        dtype=getattr(torch, args.dtype),
    )
    if args.pooling != -1:
        args_in.append(torch.arange(0, keys.numel() + 1, 2))
        expected = expected.reshape(-1, 2, 8).sum(1)
        if args.pooling == 2:
            expected /= 2
    torch.testing.assert_close(loader.run(args_in)[0], expected)
    document = {
        "paused": paused,
        "updates": updates,
        "final_cases": [
            {
                "inputs": [x.tolist() for x in args_in],
                "expected": expected.float().tolist(),
            }
        ],
    }
    (args.package_dir / "host_updates.json").write_text(json.dumps(document))
    print(
        f"Verified CPU HostLayer {args.indexer}: 32 updates, {progress[0]} concurrent AOT lookups"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument(
        "--indexer",
        choices=["identity", "fused_identity", "linear_hash_map"],
        default="fused_identity",
    )
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--pooling", type=int, choices=[-1, 1, 2], default=-1)
    parser.add_argument("--embedded", action="store_true")
    main(parser.parse_args())
