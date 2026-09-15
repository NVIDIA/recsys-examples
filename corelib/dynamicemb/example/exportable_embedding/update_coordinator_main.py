# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal local-IPC launcher for the example update coordinator."""

import argparse
import io
import os
from multiprocessing.connection import Listener

import torch


torch.ops.load_library(
    os.path.join(os.environ["DYNAMICEMB_OPS_LIB_DIR"], "inference_emb_ops.so")
)

from dynamicemb import DeltaDumpResult  # noqa: E402
from dynamicemb.exportable_embedding import (  # noqa: E402
    EmbeddingCollectionUpdateAck,
    EmbeddingCollectionUpdateCoordinator,
)


AUTHKEY = b"exportable-embedding-example"


def deserialize_delta(payload: bytes) -> DeltaDumpResult:
    value = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=True
    )
    return DeltaDumpResult(
        table_names=value["table_names"],
        keys=value["keys"],
        values=value["values"],
        evicted_keys=value["evicted_keys"],
    )


def serve(listener: Listener, coordinator: EmbeddingCollectionUpdateCoordinator) -> None:
    with listener.accept() as connection:
        while True:
            message = connection.recv()
            if message["op"] == "stop":
                return
            if message["op"] == "delta":
                update = coordinator.apply_delta(
                    message["collection_id"],
                    deserialize_delta(message["delta_payload"]),
                )
                connection.send(update.to_json())
            elif message["op"] == "ack":
                coordinator.acknowledge(
                    message["subscriber_id"],
                    EmbeddingCollectionUpdateAck.from_json(message["ack"]),
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--update-dir", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--subscriber", action="append", default=[])
    args = parser.parse_args()

    coordinator = EmbeddingCollectionUpdateCoordinator.open(
        package_dir=args.package_dir,
        shared_update_dir=args.update_dir,
        device=torch.device("cuda", torch.cuda.current_device()),
        subscriber_ids=args.subscriber,
    )
    with Listener(args.socket, family="AF_UNIX", authkey=AUTHKEY) as listener:
        serve(listener, coordinator)


if __name__ == "__main__":
    main()
