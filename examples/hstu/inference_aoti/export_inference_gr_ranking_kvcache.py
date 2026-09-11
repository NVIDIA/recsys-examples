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
import gc
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import gin
import pynve
import torch
import torch.distributed as dist
from commons.datasets import get_data_loader
from commons.datasets.hstu_batch import HSTUBatch
from commons.datasets.hstu_sequence_dataset import get_dataset
from commons.hstu_data_preprocessor import get_common_preprocessors
from commons.utils.stringify import stringify_dict
from configs import HSTUConfig, InferenceHSTUConfig, get_inference_hstu_config
from flexkv.common.config import (
    CacheConfig,
    ModelConfig,
    UserConfig,
    update_default_config_from_user_config,
)
from flexkv.server.server import KVServer
from kvcache_runtime_config import (
    config_float,
    config_int,
    config_optional_int,
    config_torch_dtype,
    extra_flexkv_configs,
    load_kvcache_runtime_yaml,
    required_config,
    reset_ipc_socket,
)
from megatron.core import parallel_state
from model import get_ranking_model
from model.export_kvcached_inference_ranking_gr import ExportKVCachedInferenceRankingGR
from modules.inference_dense_module import InferenceDenseModule
from modules.metrics import get_multi_event_metric_module
from modules.nve_compat import imported_nve_generation
from pynve.torch.nve_export import export_aot
from recsys_kvcache_manager.kvcache_config import KVCacheConfig, get_kvcache_config
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
DEFAULT_EXPORT_DIR = SCRIPT_DIR / "hstu_gr_ranking_kvcache_model"
DEFAULT_DUMP_DIR = SCRIPT_DIR / "export_test_dump"
DYNAMIC_EMBEDDING_GPU_CACHE_SIZE = 4 << 30


def start_flexkv_server(
    kvcache_config: KVCacheConfig,
    config_file: Optional[str] = None,
):
    runtime_config, resolved_config_file = load_kvcache_runtime_yaml(config_file)
    os.environ["KVCACHE_MANAGER_CONFIG_FILE"] = str(resolved_config_file)

    model_config = ModelConfig()
    cache_config = CacheConfig()
    user_config = UserConfig()

    model_config.num_layers = kvcache_config.num_layers
    model_config.num_kv_heads = kvcache_config.num_heads
    model_config.head_size = kvcache_config.head_dim
    model_config.dtype = kvcache_config.dtype
    model_config.use_mla = False
    model_config.tp_size = 1
    model_config.dp_size = 1
    cache_config.tokens_per_block = kvcache_config.page_size

    user_config.cpu_cache_gb = config_optional_int(
        runtime_config, "cpu_cache_gb"
    ) or max(
        1,
        math.ceil(
            kvcache_config.host_capacity_per_layer
            * kvcache_config.num_layers
            / (1024**3)
        ),
    )
    user_config.ssd_cache_gb = config_optional_int(runtime_config, "ssd_cache_gb") or 0
    extra_configs = kvcache_config.extra_configs or {}
    for name in (
        "enable_p2p_cpu",
        "enable_p2p_ssd",
        "enable_3rd_remote",
        "redis_host",
        "redis_port",
        "local_ip",
        "redis_password",
    ):
        if name in extra_configs:
            setattr(user_config, name, extra_configs[name])

    update_default_config_from_user_config(model_config, cache_config, user_config)

    server_recv_port = str(required_config(runtime_config, "server_recv_port"))
    gpu_register_port = str(required_config(runtime_config, "gpu_register_port"))
    reset_ipc_socket(server_recv_port)
    reset_ipc_socket(gpu_register_port)

    server_handle = KVServer.create_server(
        model_config=model_config,
        cache_config=cache_config,
        gpu_register_port=gpu_register_port,
        server_recv_port=server_recv_port,
        total_clients=1,
        inherit_env=True,
    )
    print(
        f"[INFO] Started FlexKV server for kvcache export warmup: {resolved_config_file}"
    )
    time.sleep(3)
    return server_handle


def shutdown_flexkv_runtime_and_server(model: Optional[torch.nn.Module]) -> None:
    server_handle = (
        getattr(model, "flexkv_server_handle", None) if model is not None else None
    )
    if model is not None and hasattr(model, "flexkv_server_handle"):
        model.flexkv_server_handle = None

    if hasattr(torch.ops, "kvcache_manager_ops") and hasattr(
        torch.ops.kvcache_manager_ops, "shutdown_runtime"
    ):
        torch.ops.kvcache_manager_ops.shutdown_runtime(
            torch.zeros(0, dtype=torch.int64)
        )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if server_handle is not None:
        server_handle.shutdown()


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


def make_inference_hstu_config(
    hstu_config: HSTUConfig,
    max_batch_size: int,
    max_seq_len: int,
    contextual_max_seqlen: int,
) -> InferenceHSTUConfig:
    dtype = (
        torch.bfloat16
        if hstu_config.bf16
        else torch.float16
        if hstu_config.fp16
        else torch.float32
    )
    return get_inference_hstu_config(
        hidden_size=hstu_config.hidden_size,
        num_layers=hstu_config.num_layers,
        num_attention_heads=hstu_config.num_attention_heads,
        head_dim=hstu_config.kv_channels,
        max_batch_size=max_batch_size,
        max_seq_len=math.ceil(max_seq_len / 32) * 32,
        norm_epsilon=hstu_config.layernorm_epsilon,
        dtype=dtype,
        learnable_input_layernorm=hstu_config.learnable_input_layernorm,
        residual=hstu_config.residual,
        is_causal=hstu_config.is_causal,
        target_group_size=hstu_config.target_group_size,
        position_encoding_config=hstu_config.position_encoding_config,
        contextual_max_seqlen=contextual_max_seqlen,
        scaling_seqlen=hstu_config.scaling_seqlen,
        export_mode=True,
    )


def make_export_kvcache_config(
    hstu_config: InferenceHSTUConfig,
    config_file: Optional[str] = None,
) -> KVCacheConfig:
    config, config_path = load_kvcache_runtime_yaml(config_file)
    expected_values = {
        "num_layers": hstu_config.num_layers,
        "num_kv_heads": hstu_config.num_heads,
        "head_size": hstu_config.head_dim,
        "max_batch_size": hstu_config.max_batch_size,
        "max_sequence_length": hstu_config.max_seq_len,
    }
    for name, expected in expected_values.items():
        actual = config_int(config, name)
        if actual != expected:
            raise ValueError(
                f"{config_path} has {name}={actual}, but export HSTU config expects {expected}"
            )

    expected_dtype = (
        torch.bfloat16
        if hstu_config.bf16
        else torch.float16
        if hstu_config.fp16
        else torch.float32
    )
    dtype = config_torch_dtype(config)
    if dtype != expected_dtype:
        raise ValueError(
            f"{config_path} has dtype={dtype}, but export HSTU config expects {expected_dtype}"
        )

    page_size = config_int(config, "tokens_per_page")
    offload_chunksize = config_int(config, "tokens_per_chunk")
    num_primary_cache_pages = config_int(config, "num_primary_cache_pages")
    num_heads = config_int(config, "num_kv_heads")
    head_dim = config_int(config, "head_size")
    host_capacity_per_layer = (
        config_optional_int(config, "host_capacity_per_layer")
        or num_primary_cache_pages * 2 * page_size * (num_heads * head_dim) * 2
    )
    print(f"[INFO] Loaded KV-cache runtime config: {config_path}")
    return get_kvcache_config(
        num_layers=config_int(config, "num_layers"),
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        offload_chunksize=offload_chunksize,
        num_primary_cache_pages=num_primary_cache_pages,
        num_buffer_pages=config_int(config, "num_buffer_pages"),
        host_capacity_per_layer=host_capacity_per_layer,
        max_batch_size=config_int(config, "max_batch_size"),
        max_seq_len=config_int(config, "max_sequence_length"),
        dtype=dtype,
        device=config_int(config, "device_idx"),
        host_kvstorage_backend=str(config.get("host_kvstorage_backend", "flexkv")),
        onload_timeout_ms=config_float(config, "onload_timeout_ms", 0.0),
        offload_timeout_ms=config_float(config, "offload_timeout_ms", 100.0),
        offload_mode=str(config.get("offload_mode", "lazy")),
        host_kvstorage_fail_policy=str(
            config.get("host_kvstorage_fail_policy", "fail_open")
        ),
        extra_configs=extra_flexkv_configs(config),
    )


def get_exportable_model_for_inference(
    embedding_collection_configs,
    trained_emb_table_sizes,
    checkpoint_dir,
    max_batch_size,
    total_max_seqlen,
    num_contextual_features,
    kvcache_config_file: Optional[str] = None,
):
    from dynamicemb.exportable_tables import apply_inference_embedding_collection
    from modules.exportable_embedding import apply_inference_sparse

    model = get_training_gr_model()
    model = apply_inference_embedding_collection(
        model,
        embedding_collection_configs=embedding_collection_configs,
        trained_emb_table_sizes=trained_emb_table_sizes,
    )
    inference_hstu_config = make_inference_hstu_config(
        model._hstu_config,
        max_batch_size=max_batch_size,
        max_seq_len=total_max_seqlen,
        contextual_max_seqlen=num_contextual_features,
    )
    kvcache_config = make_export_kvcache_config(
        inference_hstu_config, kvcache_config_file
    )

    sparse_module = apply_inference_sparse(model._embedding_collection)
    dense_module = InferenceDenseModule(
        inference_hstu_config,
        kvcache_config=None,
        task_config=model._task_config,
        use_cudagraph=False,
        cudagraph_configs=None,
        use_exportable=False,
        hstu_block=None,
        mlp=model._mlp,
    )

    inference_model = ExportKVCachedInferenceRankingGR(
        sparse_module=sparse_module,
        dense_module=dense_module,
        kvcache_config=kvcache_config,
    )
    inference_model.flexkv_server_handle = start_flexkv_server(
        kvcache_config, kvcache_config_file
    )
    if model._hstu_config.bf16:
        inference_model.bfloat16()
    elif model._hstu_config.fp16:
        inference_model.half()
    inference_model.load_checkpoint(checkpoint_dir)
    inference_model = inference_model.eval()
    return inference_model


def export_inference_gr_ranking(
    checkpoint_dir: str,
    max_bs: int = 1,
    stop_after_warmup: bool = False,
    kvcache_config_file: Optional[str] = None,
    export_dir_: str | os.PathLike[str] = DEFAULT_EXPORT_DIR,
    dump_dir_: str | os.PathLike[str] = DEFAULT_DUMP_DIR,
):
    export_dir, dump_dir = prepare_output_directories(export_dir_, dump_dir_)
    generation = imported_nve_generation()
    print(
        f"[INFO] Selected NVE {generation} (pynve {pynve.__version__}) from "
        f"{Path(pynve.__file__).resolve()}"
    )

    def _split_model_outputs(outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs, None
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 1:
            logits = outputs[0]
            aux = outputs[1] if len(outputs) > 1 else None
            if not isinstance(logits, torch.Tensor):
                raise TypeError(
                    f"Expected tensor logits in model outputs, got {type(logits)!r}"
                )
            return logits, aux
        raise TypeError(f"Unsupported model output type: {type(outputs)!r}")

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

    config_max_batch_size = 8
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

    model = None
    with torch.inference_mode():
        from register_hstubatch_pytree_example import register_hstu_export_pytrees

        register_hstu_export_pytrees()

        try:
            model = get_exportable_model_for_inference(
                embedding_collection_configs,
                trained_emb_table_sizes,
                checkpoint_dir,
                config_max_batch_size,
                total_max_seqlen,
                num_contextual_features,
                kvcache_config_file,
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
                total_history_lengths = (
                    (
                        torch.sum(b.features.lengths().view(-1, b.batch_size), 0).view(
                            -1
                        )
                        - b.num_candidates * 2
                    )
                    .cpu()
                    .long()
                )
                return b, user_ids, total_history_lengths

            # === Warmup ===
            batch = next(dataloader_iter)
            batch, user_ids, total_history_lengths = prepare_on_gpu(batch)
            logits, auxiliary_output = _split_model_outputs(
                model(batch, user_ids, total_history_lengths)
            )
            if auxiliary_output is not None:
                raise RuntimeError(
                    "Expected the no-offload model to return logits only"
                )
            print(f"[INFO] Warmup logits shape: {tuple(logits.shape)}")
            if stop_after_warmup:
                print("[INFO] Stopped after kvcache warmup.")
                shutdown_flexkv_runtime_and_server(model)
                return

            # === Export and Package ===
            batch = next(dataloader_iter)
            batch, user_ids, total_history_lengths = prepare_on_gpu(batch)

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
            # rebuild the HSTUBatch internally. Its batch_size is constant metadata
            # required by HSTUBatch; graph execution derives the logical batch size
            # from the input tensor shapes instead.
            class _PlainInputWrapper(torch.nn.Module):
                def __init__(self, inner, example_batch):
                    super().__init__()
                    self.inner = inner
                    self._batch_size = config_max_batch_size
                    self._keys = list(example_batch.features.keys())
                    self._contextual_feature_names = list(
                        example_batch.contextual_feature_names
                    )
                    self._item_feature_name = example_batch.item_feature_name
                    self._action_feature_name = example_batch.action_feature_name
                    self._feature_to_max_seqlen = dict(
                        example_batch.feature_to_max_seqlen
                    )
                    self._max_num_candidates = int(example_batch.max_num_candidates)
                    self._actual_batch_size = config_max_batch_size

                def _rebuild_batch(self, values, lengths, num_candidates):
                    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                        lengths.long()
                    )
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
                    user_ids,
                    total_history_lengths,
                ):
                    rebuilt = self._rebuild_batch(values, lengths, num_candidates)
                    logits, auxiliary_output = _split_model_outputs(
                        self.inner(rebuilt, user_ids.cpu(), total_history_lengths.cpu())
                    )
                    if auxiliary_output is not None:
                        raise RuntimeError(
                            "Expected the no-offload model to return logits only"
                        )
                    return logits.float().cpu()

            export_model = _PlainInputWrapper(model, batch)
            example_values = batch.features.values()
            example_lengths = batch.features.lengths()
            example_num_candidates = batch.num_candidates
            user_ids_cuda, total_history_lengths_cuda = (
                user_ids.cuda(),
                total_history_lengths.cuda(),
            )
            example_inputs = (
                example_values,
                example_lengths,
                example_num_candidates,
                user_ids_cuda,
                total_history_lengths_cuda,
            )

            with torch.inference_mode():
                rebuilt_batch = export_model._rebuild_batch(
                    example_values, example_lengths, example_num_candidates
                )

            # get dynamic shapes (now keyed on the plain tensor inputs)
            sc = ShapesCollection()
            dim_batch = Dim("batch_size", min=1, max=8)

            num_features = len(batch.features.keys())
            max_tokens = config_max_batch_size * total_max_seqlen
            sc[example_values] = {0: Dim("tokens", min=1, max=max_tokens)}
            sc[example_lengths] = {0: dim_batch * num_features}
            sc[example_num_candidates] = {0: dim_batch}
            sc[user_ids_cuda] = {0: dim_batch}
            sc[total_history_lengths_cuda] = {0: dim_batch}
            dynamic_shapes = sc.dynamic_shapes(export_model, example_inputs)
            print(f"[INFO] Dynamic shapes: {dynamic_shapes}")

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
            inputs = []
            while True:
                try:
                    batch = next(dataloader_iter)
                    batch, user_ids, total_history_lengths = prepare_on_gpu(batch)
                    inputs.append((batch, user_ids, total_history_lengths))
                except StopIteration:
                    break

            compiled_results = []
            with torch.inference_mode():
                for batch, user_ids, total_history_lengths in inputs:
                    compiled_outputs = aoti_model_runtime.run(
                        [
                            batch.features.values(),
                            batch.features.lengths(),
                            batch.num_candidates,
                            user_ids,
                            total_history_lengths,
                        ]
                    )
                    compiled_results.append(_split_model_outputs(compiled_outputs))

            with torch.inference_mode():
                for batch, user_ids, total_history_lengths in inputs:
                    logits, auxiliary_output = compiled_results[dump_idx]
                    if auxiliary_output is not None:
                        raise RuntimeError(
                            "Expected the compiled no-offload AOTI model to "
                            "return logits only"
                        )

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
                        user_ids.detach().cpu(),
                        os.path.join(dump_dir, f"batch_{dump_idx:06d}_user_ids.pt"),
                    )
                    _save_tensor_cpp_compatible(
                        total_history_lengths.detach().cpu(),
                        os.path.join(
                            dump_dir, f"batch_{dump_idx:06d}_total_history_lengths.pt"
                        ),
                    )
                    _save_tensor_cpp_compatible(
                        logits.detach().cpu(),
                        os.path.join(
                            dump_dir, f"batch_{dump_idx:06d}_compiled_logits.pt"
                        ),
                    )
                    print(f"    [Batch {dump_idx + 1}] Dumped C++ replay tensors")
                    dump_idx += 1
                    eval_module(logits.cuda(), batch.labels.values())

            print(f"[INFO] Dumped {dump_idx} test batches to {dump_dir}.")

            eval_metric_dict = eval_module.compute()
            print(
                f"[INFO][eval]:\n    "
                + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
            )

        finally:
            shutdown_flexkv_runtime_and_server(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference End-to-end Example")
    parser.add_argument("--gin_config_file", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--disable_auc", action="store_true")
    parser.add_argument("--max_bs", type=int, default=8)
    parser.add_argument(
        "--kvcache_config_file",
        type=str,
        default=None,
        help="Static YAML config shared with the C++ KV-cache runtime.",
    )
    parser.add_argument("--stop_after_warmup", action="store_true")
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
        stop_after_warmup=args.stop_after_warmup,
        kvcache_config_file=args.kvcache_config_file,
    )
    print("[INFO] Finished.")
