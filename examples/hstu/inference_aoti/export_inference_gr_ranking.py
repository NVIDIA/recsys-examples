# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import enum
import os
import sys
import warnings
from pathlib import Path

import gin
import pynve
import torch
import torch.distributed as dist
from commons.datasets import get_data_loader
from commons.datasets.hstu_batch import HSTUBatch
from commons.datasets.hstu_sequence_dataset import get_dataset
from commons.hstu_data_preprocessor import get_common_preprocessors
from commons.utils.stringify import stringify_dict
from megatron.core import parallel_state
from model import get_ranking_model
from model.inference_ranking_gr import apply_inference
from modules.metrics import get_multi_event_metric_module
from modules.nve_compat import imported_nve_generation
from pynve.torch.nve_export import export_aot
from torch.export import Dim, ShapesCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from utils import NetworkArgs, TensorModelParallelArgs

sys.path.append("./training/")
from inference_aoti.nve_aoti_compat import load_aoti, prepare_output_directories
from pretrain_gr_ranking import create_ranking_config
from trainer.utils import create_hstu_config, get_dataset_and_embedding_args

warnings.filterwarnings("default", category=UserWarning)
torch.set_warn_always(False)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPORT_DIR = SCRIPT_DIR / "hstu_gr_ranking_model"
DEFAULT_DUMP_DIR = SCRIPT_DIR / "export_test_dump"
DYNAMIC_EMBEDDING_GPU_CACHE_SIZE = 4 << 30


def init_single_rank_distributed():
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        dist.init_process_group(
            backend="gloo",  # use "nccl" only if CUDA+NCCL is properly available
            init_method="env://",
            rank=0,
            world_size=1,
        )
    parallel_state.initialize_model_parallel()


def cleanup_single_rank_distributed():
    parallel_state.destroy_model_parallel()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class RunningMode(enum.Enum):
    EVAL = "eval"
    SIMULATE = "simulate"
    EXPORT = "export"

    def __str__(self):
        return self.value


def debug_print_flattened_export_args(batch, embeddings=None) -> None:
    from torch.utils import _pytree as pytree

    print("\n===== FLATTENED EXPORT ARGS DEBUG =====")
    export_args = (batch,)
    flat_leaves, tree_spec = pytree.tree_flatten(export_args)
    print(f"Total flattened tensors: {len(flat_leaves)}")
    print(f"Tree spec: {tree_spec}\n")

    print("Batch structure:")
    batch_flat, _ = pytree.tree_flatten(batch)
    print(f"  Batch flattened count: {len(batch_flat)}")
    for i, leaf in enumerate(batch_flat):
        if isinstance(leaf, torch.Tensor):
            print(f"    [{i}] Tensor: shape={leaf.shape}, dtype={leaf.dtype}")
        else:
            print(f"    [{i}] {type(leaf).__name__}: {leaf}")

    # print(f"\nEmbeddings dict keys (order): {list(embeddings.keys())}")
    # print(f"Embeddings flattened count: {len(flat_leaves) - len(batch_flat)}")
    # embeddings_flat, _ = pytree.tree_flatten(embeddings)
    # for i, leaf in enumerate(embeddings_flat):
    #     if isinstance(leaf, torch.Tensor):
    #         print(f"    [{i}] Tensor: shape={leaf.shape}, dtype={leaf.dtype}")
    #     else:
    #         print(f"    [{i}] {type(leaf).__name__}: {leaf}")
    # print("===== END FLATTENED DEBUG =====\n")


def get_inference_dataset_and_embedding_configs(
    disable_contextual_features: bool = False,
):
    sys.path.append("./training/")
    from trainer.utils import create_embedding_configs, get_dataset_and_embedding_args

    dataset_args, embedding_args = get_dataset_and_embedding_args()
    embedding_configs = create_embedding_configs(
        dataset_args,
        NetworkArgs(),
        embedding_args,
    )

    if dataset_args.dataset_name == "kuairand-1k":
        from modules.exportable_embedding import (
            create_kuairand_embedding_collection_configs,
        )

        HASH_SIZE = 1000_064
        trained_emb_table_sizes = {
            "user_id": 1000,
            "user_active_degree": 8,
            "follow_user_num_range": 9,
            "fans_user_num_range": 9,
            "friend_user_num_range": 8,
            "register_days_range": 8,
            "video_id": HASH_SIZE,
            "action_weights": 233,
        }
        for config in embedding_configs:
            config.vocab_size = trained_emb_table_sizes[config.table_name]
        embedding_collection_configs = (
            create_kuairand_embedding_collection_configs(
                dynamic_gpu_cache_size=DYNAMIC_EMBEDDING_GPU_CACHE_SIZE
            )
        )
        return (
            dataset_args,
            embedding_configs
            if not disable_contextual_features
            else embedding_configs[-2:],
            embedding_collection_configs,
            trained_emb_table_sizes,
        )

    raise ValueError(f"dataset {dataset_args.dataset_name} is not supported")


def get_training_gr_model():
    dataset_args, embedding_args = get_dataset_and_embedding_args(False)
    network_args = NetworkArgs()

    init_single_rank_distributed()
    hstu_config = create_hstu_config(network_args, TensorModelParallelArgs())
    hstu_config.learnable_output_layernorm = False
    task_config = create_ranking_config(dataset_args, network_args, embedding_args)

    model = get_ranking_model(hstu_config=hstu_config, task_config=task_config)
    return model


def get_exportable_model_for_inference(
    embedding_collection_configs,
    trained_emb_table_sizes,
    checkpoint_dir,
):
    model = get_training_gr_model()
    inference_model = apply_inference(
        model,
        embedding_collection_configs=embedding_collection_configs,
        trained_emb_table_sizes=trained_emb_table_sizes,
        checkpoint_dir=checkpoint_dir,
    )
    return inference_model


def export_inference_gr_ranking(
    checkpoint_dir: str,
    max_bs: int = 1,
    debug_flattened_inputs: bool = False,
    export_dir_: str | os.PathLike[str] = DEFAULT_EXPORT_DIR,
    dump_dir_: str | os.PathLike[str] = DEFAULT_DUMP_DIR,
):
    export_dir, dump_dir = prepare_output_directories(export_dir_, dump_dir_)
    generation = imported_nve_generation()
    print(
        f"[INFO] Selected NVE {generation} (pynve {pynve.__version__}) from "
        f"{Path(pynve.__file__).resolve()}"
    )

    def _save_tensor_cpp_compatible(tensor: torch.Tensor, path: str) -> None:
        """Save a tensor in a format compatible with C++ torch::load().

        torch::load() expects TorchScript ZIP format, not pickle format.
        This wraps the tensor in a scripted module before saving.
        """

        class _TensorWrapper(torch.nn.Module):
            def __init__(self, t):
                super().__init__()
                self.register_buffer("tensor", t)

        wrapper = _TensorWrapper(tensor)
        torch.jit.script(wrapper).save(path)

    (
        dataset_args,
        _,
        embedding_collection_configs,
        trained_emb_table_sizes,
    ) = get_inference_dataset_and_embedding_configs()

    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = len(dataproc._contextual_feature_names)

    max_batch_size = max_bs
    total_max_seqlen = (
        dataset_args.max_num_candidates
        + dataset_args.max_history_seqlen * 2
        + num_contextual_features
    )
    print(f"[INFO] Total max sequence length: {total_max_seqlen}")

    def strip_padding_batch(batch, unpadded_batch_size):
        batch.batch_size = unpadded_batch_size
        kjt_dict = batch.features.to_dict()
        for k in kjt_dict:
            kjt_dict[k] = JaggedTensor.from_dense_lengths(
                kjt_dict[k].to_padded_dense()[: batch.batch_size],
                kjt_dict[k].lengths()[: batch.batch_size].long(),
            )
        batch.features = KeyedJaggedTensor.from_jt_dict(kjt_dict)
        batch.num_candidates = batch.num_candidates[: batch.batch_size]
        return batch

    with torch.inference_mode():
        from register_hstubatch_pytree_example import register_hstu_export_pytrees

        register_hstu_export_pytrees()

        model = get_exportable_model_for_inference(
            embedding_collection_configs,
            trained_emb_table_sizes,
            checkpoint_dir,
        )

        eval_module = get_multi_event_metric_module(
            num_classes=model.get_num_class(),
            num_tasks=model.get_num_tasks(),
            metric_types=model.get_metric_types(),
        )

        _, eval_dataset = get_dataset(
            dataset_name=dataset_args.dataset_name,
            dataset_path=dataset_args.dataset_path,
            max_history_seqlen=dataset_args.max_history_seqlen,
            max_num_candidates=dataset_args.max_num_candidates,
            num_tasks=model.get_num_tasks(),
            batch_size=max_batch_size,
            rank=0,
            world_size=1,
            shuffle=False,
            random_seed=0,
            eval_batch_size=max_batch_size,
            load_candidate_action=True,
        )

        dataloader = get_data_loader(dataset=eval_dataset)
        dataloader_iter = iter(dataloader)

        def prepare_on_gpu(b):
            b = b.to(device=torch.cuda.current_device())
            d = b.features.to_dict()
            user_ids = d["user_id"].values().cpu().long()
            if user_ids.shape[0] != b.batch_size:
                b = strip_padding_batch(b, user_ids.shape[0])
            return b

        # === Warmup ===
        batch = next(dataloader_iter)
        batch = prepare_on_gpu(batch)
        logits = model(batch)

        # === Export and Package ===
        batch = next(dataloader_iter)
        batch = prepare_on_gpu(batch)

        # preprocess batch to make it export-friendly (remove unnecessary tensors, and remove stateful tensors in KJT or JT)
        batch.features = KeyedJaggedTensor.from_lengths_sync(
            keys=batch.features.keys(),
            values=batch.features.values(),
            lengths=batch.features.lengths(),
        )
        batch.labels = None

        # ---- Plain-tuple input wrapper (Triton AOTI call_spec compatibility) ----
        # Triton's PyTorch AOTI backend accepts builtin pytree containers plus
        # tensors, but rejects custom HSTUBatch / KeyedJaggedTensor nodes in the
        # exported input spec. Export a thin wrapper with only tensor inputs and
        # rebuild the HSTUBatch internally.
        class _PlainInputWrapper(torch.nn.Module):
            def __init__(self, inner, example_batch):
                super().__init__()
                self.inner = inner
                self._batch_size = int(example_batch.batch_size)
                self._keys = list(example_batch.features.keys())
                self._contextual_feature_names = list(
                    example_batch.contextual_feature_names
                )
                self._item_feature_name = example_batch.item_feature_name
                self._action_feature_name = example_batch.action_feature_name
                self._feature_to_max_seqlen = dict(example_batch.feature_to_max_seqlen)
                self._max_num_candidates = int(example_batch.max_num_candidates)
                self._actual_batch_size = (
                    int(example_batch.actual_batch_size)
                    if example_batch.actual_batch_size is not None
                    else None
                )

            def _rebuild_batch(self, values, lengths, num_candidates):
                offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths.long())
                features = KeyedJaggedTensor(
                    keys=self._keys,
                    values=values,
                    lengths=lengths,
                    offsets=offsets,
                )
                return HSTUBatch(
                    features=features,
                    batch_size=self._batch_size,
                    feature_to_max_seqlen=self._feature_to_max_seqlen,
                    contextual_feature_names=self._contextual_feature_names,
                    actual_batch_size=self._actual_batch_size,
                    item_feature_name=self._item_feature_name,
                    action_feature_name=self._action_feature_name,
                    max_num_candidates=self._max_num_candidates,
                    num_candidates=num_candidates,
                )

            def forward(
                self,
                values,
                lengths,
                num_candidates,
            ):
                rebuilt = self._rebuild_batch(values, lengths, num_candidates)
                logits = self.inner(rebuilt)
                return logits.float().cpu()

        export_model = _PlainInputWrapper(model, batch)
        example_values = batch.features.values()
        example_lengths = batch.features.lengths()
        example_num_candidates = batch.num_candidates
        example_inputs = (
            example_values,
            example_lengths,
            example_num_candidates,
        )

        # get dynamic shapes (now keyed on the plain tensor inputs)
        sc = ShapesCollection()
        dim_batch = Dim("batch_size", min=1, max=8)

        num_features = len(batch.features.keys())
        sc[example_values] = {0: Dim("tokens", min=1, max=40000)}
        sc[example_lengths] = {0: dim_batch * num_features}
        sc[example_num_candidates] = {0: dim_batch}
        dynamic_shapes = sc.dynamic_shapes(export_model, example_inputs)
        print(f"[INFO] Dynamic shapes: {dynamic_shapes}")

        if debug_flattened_inputs:
            embeddings = None  # Placeholder for embeddings
            debug_print_flattened_export_args(batch, embeddings)

        # export & aoti_compile_and_package
        export_aot(
            export_model,
            example_inputs,
            export_dir,
            dynamic_shapes=dynamic_shapes,
        )
        print("[INFO] Exported the model package and NVE resources")

        # === Test Compiled Model ===
        aoti_model_runtime, nve_layers = load_aoti(
            export_dir,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        print(f"[INFO] Loaded {len(nve_layers)} NVE layer(s) for AOTI")

        feature_keys_dumped = False
        dump_idx = 0

        print("[INFO][check]:")
        # torch.cuda.profiler.start()
        inputs = []
        while True:
            try:
                batch = next(dataloader_iter)
                batch = prepare_on_gpu(batch)
                inputs.append(batch)

                with torch.inference_mode():
                    logits = aoti_model_runtime.run(
                        [
                            batch.features.values(),
                            batch.features.lengths(),
                            batch.num_candidates,
                        ]
                    )[0]
                    ref_logits = model(batch)
                    ref_logits_cpu = ref_logits.detach().cpu()

                    if not feature_keys_dumped:
                        torch.save(
                            list(batch.features.keys()),
                            os.path.join(dump_dir, "feature_keys.pt"),
                        )
                        feature_keys_dumped = True

                    _save_tensor_cpp_compatible(
                        batch.features.values().detach().cpu(),
                        os.path.join(dump_dir, f"batch_{dump_idx:06d}_values.pt"),
                    )
                    _save_tensor_cpp_compatible(
                        batch.features.lengths().detach().cpu(),
                        os.path.join(dump_dir, f"batch_{dump_idx:06d}_lengths.pt"),
                    )
                    _save_tensor_cpp_compatible(
                        batch.num_candidates.detach().cpu(),
                        os.path.join(
                            dump_dir, f"batch_{dump_idx:06d}_num_candidates.pt"
                        ),
                    )
                    _save_tensor_cpp_compatible(
                        ref_logits_cpu,
                        os.path.join(dump_dir, f"batch_{dump_idx:06d}_ref_logits.pt"),
                    )
                    dump_idx += 1

                    print(
                        f"    [Batch {dump_idx}] Check equal:",
                        torch.max(torch.abs(logits - ref_logits_cpu)).item() <= 0.0625,
                    )

                eval_module(logits.cuda(), batch.labels.values())
            except StopIteration:
                break
        # torch.cuda.profiler.stop()

        print(f"[INFO] Dumped {dump_idx} test batches to {dump_dir}.")

        eval_metric_dict = eval_module.compute()
        print(
            f"[INFO][eval]:\n    "
            + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
        )

        # === Benchmark Compiled Model ===
        print("[INFO][benchmark]:")
        print(
            "    Benchmark on GPU:",
            torch.cuda.get_device_name(torch.cuda.current_device()),
        )
        num_benchmark_batches = len(inputs)
        num_logical_requests = sum(int(b.batch_size) for b in inputs)
        if num_benchmark_batches == 0 or num_logical_requests == 0:
            raise RuntimeError("No inputs were collected for benchmarking")
        print(
            "    Benchmark workload: "
            f"{num_benchmark_batches} batches, "
            f"{num_logical_requests} logical requests"
        )
        import time

        compiled_time = []
        for _ in range(3):
            torch.cuda.synchronize()
            results = []
            start = time.perf_counter()
            with torch.inference_mode():
                for b in inputs:
                    logits = aoti_model_runtime.run(
                        [
                            b.features.values(),
                            b.features.lengths(),
                            b.num_candidates,
                        ]
                    )[0]
                    results.append(logits)
            torch.cuda.synchronize()
            end = time.perf_counter()
            compiled_time.append(end - start)
        compiled_time_avg = sum(compiled_time) / len(compiled_time)
        print(
            f"    Compiled model elapsed time: {compiled_time_avg:.6f} seconds; "
            f"{compiled_time_avg * 1000.0 / num_benchmark_batches:.3f} "
            "ms/batch; "
            f"{compiled_time_avg * 1000.0 / num_logical_requests:.3f} "
            "ms/logical request"
        )

        for item in results:
            del item

        import time

        python_time = []
        for _ in range(3):
            torch.cuda.synchronize()
            results = []
            start = time.perf_counter()
            with torch.inference_mode():
                for b in inputs:
                    ref_logits = model(b)
                    results.append(ref_logits)
            torch.cuda.synchronize()
            end = time.perf_counter()
            python_time.append(end - start)
        python_time_avg = sum(python_time) / len(python_time)
        print(
            f"    Python model elapsed time: {python_time_avg:.6f} seconds; "
            f"{python_time_avg * 1000.0 / num_benchmark_batches:.3f} "
            "ms/batch; "
            f"{python_time_avg * 1000.0 / num_logical_requests:.3f} "
            "ms/logical request"
        )

        for item in results:
            del item


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference End-to-end Example")
    parser.add_argument("--gin_config_file", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--disable_auc", action="store_true")
    parser.add_argument("--max_bs", type=int, default=2)
    parser.add_argument("--debug_flattened_inputs", action="store_true")
    parser.add_argument(
        "--export_dir",
        type=str,
        default=str(DEFAULT_EXPORT_DIR),
        help="Model export directory (default: %(default)s).",
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        default=str(DEFAULT_DUMP_DIR),
        help="Replay dump directory (default: %(default)s).",
    )

    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)

    if args.max_bs <= 1:
        print(
            "[WARNING] Max batch size (max_bs) is set to 1, which causes the torch compiler fails to capture the dynamic shapes.\n"
            "          Adjusted max_bs to 2 for successful export."
        )
        args.max_bs = 2

    export_inference_gr_ranking(
        checkpoint_dir=args.checkpoint_dir,
        export_dir_=args.export_dir,
        dump_dir_=args.dump_dir,
        max_bs=args.max_bs,
        debug_flattened_inputs=args.debug_flattened_inputs,
    )
    print("[INFO] Finished.")
