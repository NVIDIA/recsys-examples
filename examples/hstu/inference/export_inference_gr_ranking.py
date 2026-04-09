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
import math
import os
import sys
import warnings

import gin
import torch
from commons.datasets import get_data_loader
from commons.datasets.hstu_sequence_dataset import get_dataset
from commons.hstu_data_preprocessor import get_common_preprocessors
from commons.utils.stringify import stringify_dict
from configs import (
    InferenceEmbeddingConfig,
    PositionEncodingConfig,
    RankingConfig,
    get_hstu_config,
)
from modules.metrics import get_multi_event_metric_module
from modules.inference_dense_module import get_inference_dense_model
from modules.exportable_embedding import ExportableEmbedding
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from utils import DatasetArgs, NetworkArgs, RankingArgs

sys.path.append("./model/")
from inference_ranking_gr import InferenceRankingGR
from torch.export import Dim, ExportedProgram, ShapesCollection


# Reduce warning spam: show each UserWarning once per source location.
warnings.filterwarnings("default", category=UserWarning)
torch.set_warn_always(False)


class RunningMode(enum.Enum):
    EVAL = "eval"
    SIMULATE = "simulate"
    EXPORT = "export"

    def __str__(self):
        return self.value


def debug_print_flattened_export_args(batch, embeddings = None) -> None:
    from torch.utils import _pytree as pytree

    print("\n===== FLATTENED EXPORT ARGS DEBUG =====")
    export_args = (batch, )
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
    dataset_args = DatasetArgs()
    embedding_dim = NetworkArgs().hidden_size
    HASH_SIZE = 1000_064
    if dataset_args.dataset_name == "kuairand-1k":
        embedding_configs = [
            InferenceEmbeddingConfig(
                feature_names=["user_id"],
                table_name="user_id",
                vocab_size=1000,
                dim=embedding_dim,
                use_dynamicemb=True,
            ),
            InferenceEmbeddingConfig(
                feature_names=["user_active_degree"],
                table_name="user_active_degree",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["follow_user_num_range"],
                table_name="follow_user_num_range",
                vocab_size=9,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["fans_user_num_range"],
                table_name="fans_user_num_range",
                vocab_size=9,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["friend_user_num_range"],
                table_name="friend_user_num_range",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["register_days_range"],
                table_name="register_days_range",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["video_id"],
                table_name="video_id",
                vocab_size=HASH_SIZE,
                dim=embedding_dim,
                use_dynamicemb=True,
            ),
            InferenceEmbeddingConfig(
                feature_names=["action_weights"],
                table_name="action_weights",
                vocab_size=256,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
        ]
        return (
            dataset_args,
            embedding_configs
            if not disable_contextual_features
            else embedding_configs[-2:],
        )

    raise ValueError(f"dataset {dataset_args.dataset_name} is not supported")


def get_inference_export_model(
    emb_configs,
    max_batch_size,
    num_contextual_features,
    total_max_seqlen,
    checkpoint_dir,
):
    network_args = NetworkArgs()
    if network_args.dtype_str == "bfloat16":
        inference_dtype = torch.bfloat16
    # elif network_args.dtype_str == "float16":
    #     inference_dtype = torch.float16
    else:
        raise ValueError(
            f"Inference data type {network_args.dtype_str} is not supported"
        )

    position_encoding_config = PositionEncodingConfig(
        num_position_buckets=8192,
        num_time_buckets=2048,
        use_time_encoding=False,
        static_max_seq_len=math.ceil(total_max_seqlen / 32) * 32,
    )

    hstu_config = get_hstu_config(
        hidden_size=network_args.hidden_size,
        kv_channels=network_args.kv_channels,
        num_attention_heads=network_args.num_attention_heads,
        num_layers=network_args.num_layers,
        dtype=inference_dtype,
        position_encoding_config=position_encoding_config,
        learnable_input_layernorm = True,
        learnable_output_layernorm = False,
        is_inference = True,
    )

    ranking_args = RankingArgs()
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )


    sparse_module = ExportableEmbedding(emb_configs)
    sparse_module.eval()

    dense_module = get_inference_dense_model(
        hstu_config,
        None,  # kvcache_config is not needed for export
        task_config,
        use_exportable=True,
    )
    if hstu_config.bf16:
        dense_module.bfloat16()
    elif hstu_config.fp16:
        dense_module.half()
    dense_module.eval()

    model = InferenceRankingGR(
        sparse_module,
        dense_module,
    )
    if hstu_config.bf16:
        model.bfloat16()
    elif hstu_config.fp16:
        model.half()
    model.load_checkpoint(checkpoint_dir)
    model.eval()

    return model


def get_configs(
    emb_configs,
    num_contextual_features,
    total_max_seqlen,
):
    network_args = NetworkArgs()
    if network_args.dtype_str == "bfloat16":
        inference_dtype = torch.bfloat16
    # elif network_args.dtype_str == "float16":
    #     inference_dtype = torch.float16
    else:
        raise ValueError(
            f"Inference data type {network_args.dtype_str} is not supported"
        )

    position_encoding_config = PositionEncodingConfig(
        num_position_buckets=8192,
        num_time_buckets=2048,
        use_time_encoding=False,
        static_max_seq_len=math.ceil(total_max_seqlen / 32) * 32,
    )

    hstu_config = get_hstu_config(
        hidden_size=network_args.hidden_size,
        kv_channels=network_args.kv_channels,
        num_attention_heads=network_args.num_attention_heads,
        num_layers=network_args.num_layers,
        dtype=inference_dtype,
        learnable_input_layernorm = False,
        learnable_output_layernorm = False,
        is_inference = True,
    )

    ranking_args = RankingArgs()
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )

    return hstu_config, task_config


def export_inference_gr_ranking(
    checkpoint_dir: str,
    max_bs: int = 1,
    debug_flattened_inputs: bool = False,
):
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
    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs()

    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = len(dataproc._contextual_feature_names)

    max_batch_size = max_bs
    total_max_seqlen = dataset_args.max_num_candidates + dataset_args.max_history_seqlen * 2 + num_contextual_features
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
    
    hstu_config, task_config = get_configs(
        emb_configs,
        num_contextual_features,
        total_max_seqlen,
    )
    

    with torch.inference_mode():
        from register_hstubatch_pytree_example import register_hstu_export_pytrees
        register_hstu_export_pytrees()

        model = get_inference_export_model(
            emb_configs,
            max_batch_size,
            num_contextual_features,
            total_max_seqlen,
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
            load_candidate_action=False,
        )

        dataloader = get_data_loader(dataset=eval_dataset)
        dataloader_iter = iter(dataloader)

        def strip_padding_batch(b):
            b = b.to(device=torch.cuda.current_device())
            d = b.features.to_dict()
            user_ids = d["user_id"].values().cpu().long()
            if user_ids.shape[0] != b.batch_size:
                b = strip_padding_batch(b, user_ids.shape[0])
            return b

        # === Warmup ===
        batch = next(dataloader_iter)
        batch = strip_padding_batch(batch)
        logits = model(batch)

        # === Export and Package ===
        batch = next(dataloader_iter)
        batch = strip_padding_batch(batch)

        # preprocess batch to make it export-friendly (remove unnecessary tensors, and remove stateful tensors in KJT or JT)
        batch.features = KeyedJaggedTensor.from_lengths_sync(
            keys=batch.features.keys(),
            values=batch.features.values(),
            lengths=batch.features.lengths(),
        )
        batch.labels = None
        
        # get dynamic shapes
        sc = ShapesCollection()
        # dim_batch = Dim("batch", min=8 , max=4 * 8)

        sc[batch.features.values()] = {0: Dim.AUTO}
        sc[batch.features.lengths()] = {0: Dim.AUTO}
        sc[batch.num_candidates] = {0: Dim.AUTO}
        dynamic_shapes = sc.dynamic_shapes(model, (batch,))
        print(f"[INFO] Dynamic shapes: {dynamic_shapes}")

        if debug_flattened_inputs:
            debug_print_flattened_export_args(batch, embeddings)

        # export & aoti_compile_and_package
        exported_program: ExportedProgram = torch.export.export(
            model, args=(batch,), dynamic_shapes=dynamic_shapes,
        )   
        _ = torch._inductor.aoti_compile_and_package(
            exported_program,
            package_path=os.path.join(os.path.dirname(__file__), "hstu_gr_ranking_model.pt2")
        )
        print(f"[INFO] Exported and packaged the model to {os.path.join(os.path.dirname(__file__), 'hstu_gr_ranking_model.pt2')}.")


        # === Test Exported Model ===
        dump_dir = os.path.join(os.path.dirname(__file__), "export_test_dump")
        os.makedirs(dump_dir, exist_ok=True)
        feature_keys_dumped = False
        dump_idx = 0

        # torch.cuda.profiler.start()
        print(batch.features.values().shape)
        while True:
            try:
                batch = next(dataloader_iter)

                batch = batch.to(device=torch.cuda.current_device())
                d = batch.features.to_dict()
                user_ids = d["user_id"].values().cpu().long()
                if user_ids.shape[0] != batch.batch_size:
                    batch = strip_padding_batch(batch, user_ids.shape[0])
                if batch.features.values().size(0) != total_max_seqlen:
                    continue

                with torch.inference_mode():
                    torch.cuda.nvtx.range_push("HSTU embedding")
                    # embeddings = model.sparse_module(batch.features)
                    torch.cuda.nvtx.range_pop()
                    logits = exported_program.module()(batch)

                    ref_logits = model(batch)

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
                        os.path.join(dump_dir, f"batch_{dump_idx:06d}_num_candidates.pt"),
                    )
                    _save_tensor_cpp_compatible(
                        ref_logits.detach().cpu(),
                        os.path.join(dump_dir, f"batch_{dump_idx:06d}_ref_logits.pt"),
                    )
                    dump_idx += 1

                    print(f"[Batch {dump_idx}] Check equal:", torch.allclose(logits, ref_logits))

                eval_module(logits, batch.labels.values())
            except StopIteration:
                break
        # torch.cuda.profiler.stop()

        print(f"[INFO] Dumped {dump_idx} test batches to {dump_dir}.")

        eval_metric_dict = eval_module.compute()
        print(
            f"[eval]:\n    "
            + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference End-to-end Example")
    parser.add_argument("--gin_config_file", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--disable_auc", action="store_true")
    parser.add_argument("--max_bs", type=int, default=1)
    parser.add_argument("--debug_flattened_inputs", action="store_true")

    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)

    export_inference_gr_ranking(
        checkpoint_dir=args.checkpoint_dir,
        max_bs=args.max_bs,
        debug_flattened_inputs=args.debug_flattened_inputs,
    )
    print("Finished.")
