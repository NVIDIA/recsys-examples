"""Dry-run inspect a Qwen3 HuggingFace checkpoint for GR inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tool_utils import bootstrap_repo_paths

bootstrap_repo_paths(__file__)

from gr_inference.gr_models import HFCheckpointLoader
from gr_inference.gr_models.qwen3 import Qwen3HFAdapter, identify_qwen3_variant


def build_report(model_dir: str | Path, *, max_layers: int = 2) -> dict:
    manifest = HFCheckpointLoader(model_dir).manifest()
    adapter = Qwen3HFAdapter.from_manifest(manifest)
    variant = identify_qwen3_variant(adapter.config)
    plan = adapter.load_plan()
    plan.validate(manifest)
    grouped = plan.grouped_by_file(manifest)

    layer_requests = [
        request
        for request in plan.requests
        if request.logical_name.startswith("layers.")
    ]
    shown_layer_requests = [
        request
        for request in layer_requests
        if int(request.logical_name.split(".")[1]) < max_layers
    ]

    return {
        "model_dir": str(manifest.model_dir),
        "model_name": adapter.config.model_name,
        "known_variant": variant.canonical_name if variant is not None else None,
        "num_layers": adapter.config.num_layers,
        "hidden_size": adapter.config.hidden_size,
        "num_attention_heads": adapter.config.num_attention_heads,
        "num_kv_heads": adapter.config.num_kv_heads,
        "head_dim": adapter.config.head_dim,
        "q_size": adapter.config.q_size,
        "kv_size": adapter.config.kv_size,
        "qkv_size": adapter.config.qkv_size,
        "intermediate_size": adapter.config.intermediate_size,
        "gate_up_size": adapter.config.gate_up_size,
        "vocab_size": adapter.config.vocab_size,
        "tie_word_embeddings": adapter.config.tie_word_embeddings,
        "weight_files": list(manifest.weight_files),
        "num_tensors": len(manifest.tensor_map),
        "num_load_requests": len(plan.requests),
        "grouped_files": {
            filename: len(tensor_names)
            for filename, tensor_names in grouped.items()
        },
        "sample_requests": [
            {
                "logical_name": request.logical_name,
                "source_names": list(request.source_names),
                "transform": request.transform,
                "dim": request.dim,
                "required": request.required,
            }
            for request in shown_layer_requests
        ],
    }


def print_text_report(report: dict) -> None:
    print("Qwen3 checkpoint dry-run")
    print("=" * 72)
    print(f"model_dir: {report['model_dir']}")
    print(f"model_name: {report['model_name']}")
    print(f"known_variant: {report['known_variant']}")
    print(
        "shape: "
        f"layers={report['num_layers']} hidden={report['hidden_size']} "
        f"Hq={report['num_attention_heads']} Hkv={report['num_kv_heads']} "
        f"D={report['head_dim']} intermediate={report['intermediate_size']} "
        f"vocab={report['vocab_size']} tied={report['tie_word_embeddings']}"
    )
    print(
        "packed: "
        f"q={report['q_size']} kv={report['kv_size']} "
        f"qkv={report['qkv_size']} gate_up={report['gate_up_size']}"
    )
    print(
        f"weights: files={len(report['weight_files'])} "
        f"tensors={report['num_tensors']} load_requests={report['num_load_requests']}"
    )
    print("\nGrouped required tensors by file:")
    for filename, count in report["grouped_files"].items():
        print(f"  {filename}: {count}")
    print("\nSample layer load requests:")
    for request in report["sample_requests"]:
        sources = ", ".join(request["source_names"])
        print(
            f"  {request['logical_name']} <- {sources} "
            f"({request['transform']} dim={request['dim']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-layers", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report(args.model_dir, max_layers=args.max_layers)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)


if __name__ == "__main__":
    main()
