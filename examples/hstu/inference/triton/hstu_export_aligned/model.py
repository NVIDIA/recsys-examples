# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton Python backend using the same model conversion as the AOTI exporter."""

import json
import sys
from pathlib import Path

import gin
import torch
import triton_python_backend_utils as pb_utils


def _parameter(model_config, name):
    try:
        return model_config["parameters"][name]["string_value"]
    except KeyError as error:
        raise ValueError(f"Missing required Triton model parameter: {name}") from error


class TritonPythonModel:
    def initialize(self, args):
        self._model_config = json.loads(args["model_config"])
        self._device = torch.device(f"cuda:{args['model_instance_device_id']}")
        torch.cuda.set_device(self._device)

        hstu_root = Path(_parameter(self._model_config, "HSTU_ROOT")).resolve()
        gin_config = Path(
            _parameter(self._model_config, "HSTU_GIN_CONFIG_FILE")
        ).resolve()
        checkpoint_dir = Path(
            _parameter(self._model_config, "HSTU_CHECKPOINT_DIR")
        ).resolve()
        self._max_batch_size = int(
            _parameter(self._model_config, "HSTU_MAX_BATCH_SIZE")
        )

        if self._max_batch_size < 1 or self._max_batch_size > 8:
            raise ValueError(
                "HSTU_MAX_BATCH_SIZE must be in [1, 8], got " f"{self._max_batch_size}"
            )
        for required_path in (hstu_root, gin_config, checkpoint_dir):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Required path does not exist: {required_path}"
                )

        # The exporter is written as an examples/hstu entry point, so make its
        # package and training imports deterministic inside the Python stub.
        for import_root in (
            hstu_root.parent,
            hstu_root,
            hstu_root / "training",
        ):
            import_root_string = str(import_root)
            if import_root_string not in sys.path:
                sys.path.insert(0, import_root_string)

        from commons.datasets.hstu_batch import HSTUBatch
        from commons.hstu_data_preprocessor import get_common_preprocessors
        from inference_aoti.export_inference_gr_ranking import (
            cleanup_single_rank_distributed,
            get_exportable_model_for_inference,
            get_inference_dataset_and_embedding_configs,
        )
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        # Importing the exporter registers the Gin-configurable HSTU classes.
        gin.parse_config_file(str(gin_config))

        (
            dataset_args,
            _,
            embedding_collection_configs,
            trained_emb_table_sizes,
        ) = get_inference_dataset_and_embedding_configs()
        data_processor = get_common_preprocessors("")[dataset_args.dataset_name]

        self._keys = list(data_processor._contextual_feature_names) + [
            data_processor._item_feature_name,
            data_processor._action_feature_name,
        ]
        self._contextual_feature_names = list(data_processor._contextual_feature_names)
        self._item_feature_name = data_processor._item_feature_name
        self._action_feature_name = data_processor._action_feature_name
        self._max_num_candidates = int(dataset_args.max_num_candidates)
        sequence_feature_max_length = (
            int(dataset_args.max_history_seqlen) + self._max_num_candidates
        )
        self._feature_to_max_seqlen = {
            **{name: 1 for name in self._contextual_feature_names},
            self._item_feature_name: sequence_feature_max_length,
            self._action_feature_name: sequence_feature_max_length,
        }

        self._hstu_batch_type = HSTUBatch
        self._kjt_type = KeyedJaggedTensor
        self._cleanup_distributed = cleanup_single_rank_distributed

        with torch.inference_mode():
            self._model = get_exportable_model_for_inference(
                embedding_collection_configs,
                trained_emb_table_sizes,
                str(checkpoint_dir),
            )

        output_config = pb_utils.get_output_config_by_name(
            self._model_config, "OUTPUT__0"
        )
        if output_config["data_type"] != "TYPE_FP32":
            raise ValueError("OUTPUT__0 must use TYPE_FP32")

        print(
            "[hstu_export_aligned] initialized "
            f"{len(self._keys)} features on {self._device}; "
            f"maximum logical batch size {self._max_batch_size}; "
            "model factory=inference_aoti.export_inference_gr_ranking."
            "get_exportable_model_for_inference; KV cache=disabled"
        )

    def _make_batch(self, values, lengths, num_candidates, batch_size):
        features = self._kjt_type.from_lengths_sync(
            keys=self._keys,
            values=values,
            lengths=lengths,
        )
        return self._hstu_batch_type(
            features=features,
            batch_size=batch_size,
            feature_to_max_seqlen=self._feature_to_max_seqlen,
            contextual_feature_names=self._contextual_feature_names,
            actual_batch_size=batch_size,
            item_feature_name=self._item_feature_name,
            action_feature_name=self._action_feature_name,
            max_num_candidates=self._max_num_candidates,
            num_candidates=num_candidates,
        )

    def _validate_numpy_inputs(self, values, lengths, num_candidates):
        if values.ndim != 1 or lengths.ndim != 1 or num_candidates.ndim != 1:
            raise ValueError("All HSTU inputs must be one-dimensional")
        batch_size = int(num_candidates.size)
        if batch_size < 1 or batch_size > self._max_batch_size:
            raise ValueError(
                f"Logical batch size {batch_size} is outside "
                f"[1, {self._max_batch_size}]"
            )
        expected_lengths = len(self._keys) * batch_size
        if int(lengths.size) != expected_lengths:
            raise ValueError(
                f"INPUT__1 has {int(lengths.size)} lengths; expected "
                f"{expected_lengths} for {len(self._keys)} features and "
                f"batch size {batch_size}"
            )
        if (lengths < 0).any():
            raise ValueError("INPUT__1 contains a negative feature length")
        if int(lengths.sum()) != int(values.size):
            raise ValueError("INPUT__1 lengths do not sum to INPUT__0 size")
        if (num_candidates < 0).any():
            raise ValueError("INPUT__2 contains a negative candidate count")
        if (num_candidates > self._max_num_candidates).any():
            raise ValueError(
                f"INPUT__2 exceeds max_num_candidates={self._max_num_candidates}"
            )
        return batch_size

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                values_numpy = pb_utils.get_input_tensor_by_name(
                    request, "INPUT__0"
                ).as_numpy()
                lengths_numpy = pb_utils.get_input_tensor_by_name(
                    request, "INPUT__1"
                ).as_numpy()
                num_candidates_numpy = pb_utils.get_input_tensor_by_name(
                    request, "INPUT__2"
                ).as_numpy()
                batch_size = self._validate_numpy_inputs(
                    values_numpy, lengths_numpy, num_candidates_numpy
                )

                values = torch.from_numpy(values_numpy).to(
                    device=self._device, dtype=torch.int64
                )
                lengths = torch.from_numpy(lengths_numpy).to(
                    device=self._device, dtype=torch.int64
                )
                num_candidates = torch.from_numpy(num_candidates_numpy).to(
                    device=self._device, dtype=torch.int64
                )

                batch = self._make_batch(values, lengths, num_candidates, batch_size)
                with torch.inference_mode():
                    logits = self._model(batch).float().cpu().numpy()
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("OUTPUT__0", logits)]
                    )
                )
            except Exception as error:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            f"hstu_export_aligned inference failed: {error}"
                        )
                    )
                )
        return responses

    def finalize(self):
        print("[hstu_export_aligned] cleaning up distributed state")
        if hasattr(self, "_cleanup_distributed"):
            try:
                self._cleanup_distributed()
            except Exception as error:
                print(f"[hstu_export_aligned] distributed cleanup warning: {error}")
        gin.clear_config()
