#!/usr/bin/env python3
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
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import logging
from collections import deque
from typing import (
    Any,
    Callable,
    Deque,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    cast,
)

import nvtx
import torch
from commons.utils.distributed_utils import collective_assert
from distributed.finalize_model_grads import finalize_model_grads
from megatron.core import parallel_state
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from pipeline.utils import (
    In,
    Out,
    PipelinedForward,
    PipelinedPostproc,
    PrefetchPipelinedForward,
    PrefetchTrainPipelineContext,
    TrainPipelineContext,
    _override_input_dist_forwards,
    _pipeline_detach_model,
    _prefetch_embeddings,
    _rewrite_model,
    _start_data_dist,
    _to_device,
    _wait_for_batch,
)
from torch.autograd.profiler import record_function
from torchrec.distributed.dist_data import KJTAllToAllTensorsAwaitable
from torchrec.distributed.model_parallel import ShardedModule
from torchrec.distributed.types import Awaitable
from torchrec.pt2.checks import is_torchdynamo_compiling
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)
# This is required to support older torch package export for older models
try:
    pass
except ImportError:
    logger.warning("torchrec_use_sync_collectives is not available")

if not torch._running_with_deploy():
    torch.ops.import_module("fbgemm_gpu.sparse_ops")


class TrainPipeline(abc.ABC, Generic[In, Out]):
    @abc.abstractmethod
    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        pass


class TrainPipelineSparseDist(TrainPipeline[In, Out]):
    """
    This pipeline overlaps device transfer, and `ShardedModule.input_dist()` with
    forward and backward. This helps hide the all2all latency while preserving the
    training forward / backward ordering.

    stage 3: forward, backward - uses default CUDA stream
    stage 2: ShardedModule.input_dist() - uses data_dist CUDA stream
    stage 1: device transfer - uses memcpy CUDA stream

    `ShardedModule.input_dist()` is only done for top-level modules in the call graph.
    To be considered a top-level module, a module can only depend on 'getattr' calls on
    input.

    Input model must be symbolically traceable with the exception of `ShardedModule` and
    `DistributedDataParallel` modules.

    Args:
        model (torch.nn.Module): model to pipeline.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device where device transfer, sparse data dist, and
            forward/backward pass will happen.
        execute_all_batches (bool): executes remaining batches in pipeline after
            exhausting dataloader iterator.
        apply_jit (bool): apply torch.jit.script to non-pipelined (unsharded) modules.
    """

    # The PipelinedForward class that is used in _rewrite_model
    _pipelined_forward_type = PipelinedForward

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        context_type: Type[TrainPipelineContext] = TrainPipelineContext,
        # keep for backward compatibility
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._device = device
        self._execute_all_batches = execute_all_batches
        self._apply_jit = apply_jit

        if device.type == "cuda":
            # use two data streams to support two concurrent batches
            # Dynamo does not support cuda stream specification,
            # this freedom is left for compiler pipelining optimizations.
            assert (
                not is_torchdynamo_compiling()
            ), "Train Pipelines rely on cuda streams, which is not supported by Dynamo"

        # pyre-ignore
        self._stream_context = (
            torch.get_device_module(self._device).stream
            if self._device.type in ["cuda", "mtia"]
            else torch.cuda.stream
        )

        self._memcpy_stream: Optional[torch.Stream] = (
            (torch.get_device_module(device).Stream(priority=-1))
            if device.type in ["cuda", "mtia"]
            else None
        )
        self._data_dist_stream: Optional[torch.Stream] = (
            (torch.get_device_module(device).Stream(priority=-1))
            if device.type in ["cuda", "mtia"]
            else None
        )

        # pyre-ignore
        self._original_forwards: List[Callable[..., Any]] = []

        self._original_kjt_dist_forwards: List[
            Callable[[KeyedJaggedTensor], Awaitable[KJTAllToAllTensorsAwaitable]]
        ] = []

        self._model_attached = True
        self._pipeline_postproc = pipeline_postproc

        self._next_index: int = 0
        self.contexts: Deque[TrainPipelineContext] = deque()
        self._pipelined_modules: List[ShardedModule] = []
        self._pipelined_postprocs: List[PipelinedPostproc] = []
        self.batches: Deque[Optional[In]] = deque()
        self._dataloader_iter: Optional[Iterator[In]] = None
        self._dataloader_exhausted: bool = False
        self._context_type: Type[TrainPipelineContext] = context_type

        self._model_fwd: Callable[[Optional[In]], Tuple[torch.Tensor, Out]] = (
            custom_model_fwd if custom_model_fwd else model
        )

        # DEPRECATED FIELDS
        self._batch_i: Optional[In] = None
        self._batch_ip1: Optional[In] = None
        self._batch_ip2: Optional[In] = None
        self._context: TrainPipelineContext = context_type(version=0)

    def detach(self) -> torch.nn.Module:
        """
        Detaches the model from sparse data dist (SDD) pipeline. A user might want to get
        the original model back after training. The original model.forward was previously
        modified by the train pipeline. for more please see:
        https://github.com/pytorch/torchrec/pull/2076

        To use the pipeline after detaching the model, pipeline.attach(model)
        needs to be called.
        Inflight batches are kept so pipeline.progress(data_iter) can be resumed normally.

        Returns the original model.
        """
        if self._pipelined_modules:
            _pipeline_detach_model(
                model=self._model,
                pipelined_modules=self._pipelined_modules,
                original_forwards=self._original_forwards,
                original_kjt_dist_forwards=self._original_kjt_dist_forwards,
                pipelined_postprocs=self._pipelined_postprocs,
            )

        self._model_attached = False
        return self._model

    def attach(
        self, model: Optional[torch.nn.Module] = None, sparse_dist: bool = True
    ) -> None:
        """
        should be used with detach function. these functions should only be used from user code,
        when user want to switch the train pipeline. for more please see:
        https://github.com/pytorch/torchrec/pull/2076
        """
        if model:
            self._model = model

        self._model_attached = True
        if self.contexts:
            self._pipeline_model(
                batch=self.batches[0] if sparse_dist else None,
                context=self.contexts[0],
                pipelined_forward=self._pipelined_forward_type,
            )
        else:
            # attaching the model after end of train pipeline
            # model rewrite for SDD needs context but self.contexts is empty
            # reset _pipelined_modules so _fill_pipeline will rewrite model on progress()
            self._pipelined_modules = []
            self._pipelined_postprocs = []

    def _set_module_context(self, context: TrainPipelineContext) -> None:
        """
        pipelined modules are the TorchRec's sparse modules like shardedEBC, shardedEC, etc.
        the forward function is swapped with a PipelinedForward in the _rewrite_model call.
        The PipelinedForward needs a context to correctly perform the forward behavior.
        please check PipelinedForward for details.
        """
        for module in self._pipelined_modules:
            module.forward.set_context(context)

        for postproc_module in self._pipelined_postprocs:
            # This ensures that next iter model fwd uses cached results
            postproc_module.set_context(context)

    def enqueue_batch(self, dataloader_iter: Iterator[In]) -> bool:
        """
        load a data batch from dataloader, and copy it from cpu to gpu
        also create the context for this batch.
        """
        batch, context = self.copy_batch_to_gpu(dataloader_iter)
        if batch is None:
            return False
        self.batches.append(batch)
        # pyre-ignore [6]
        self.contexts.append(context)

        return True

    def dequeue_batch(self) -> None:
        """
        remove a processed batch from the batch queue, also set the module context if applicable
        """
        self.batches.popleft()
        self.contexts.popleft()

        # update PipelinedForward context to match next forward pass
        if len(self.batches) >= 1:
            self._set_module_context(self.contexts[0])

    def fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        """
        This function is called in self.progress (one of the main APIs for running train pipeline)
        Here we assume the max pipelined len(batches) == 2 (capacity), which will be the most common
        scenario during the full training job, when this function is effectively doing nothing.
        There would only be two other scenarios:
        len(batches) == 0:
            initialize the pipeline, fill in two batches, start input_dist for the first batch.
        len(batches) == 1:
            dataloader_iter stops, the last batch, do nothing
        """

        # pipeline is already filled with max capacity (2)
        if len(self.batches) >= 2:
            return

        # executes last batch in pipeline, when there is only one batch in the pipeline
        # TODO: this _execute_all_batches doesn't really work here D43546239. it will
        # just throw an exception at copy_to_gpu when the dataloader is exhausted
        if self.batches and self._execute_all_batches:
            return

        # batch i, data (batch) and context
        if not self.enqueue_batch(dataloader_iter):
            return

        # modify the (sharded) sparse module forward, and invoke the first part of input_dist
        self._init_pipelined_modules(
            # pyre-ignore [6]
            self.batches[0],
            self.contexts[0],
            self._pipelined_forward_type,
        )
        # doing the second part of input_dist, the first part is invoked in _init_pipelined_modules
        self.wait_sparse_data_dist(self.contexts[0])

        # batch i+1
        if not self.enqueue_batch(dataloader_iter):
            return

    def _wait_for_batch(self) -> None:
        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self.batches[0]), self._data_dist_stream)

    def _backward(self, losses: torch.Tensor) -> None:
        with record_function("## backward ##"):
            torch.sum(losses, dim=0).backward()

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        """
        For TrainPipelineSparseDist, we assume the max pipelined batches == 3 (capacity):
            batches[0]: current batch, for emb_lookup, output_dist, and fwd/bwd/opt (expecting input_dist)
            batches[1]: next batch, for input_dist (expecting copied to device)
            batches[2]: i+2 batch, for copy_batch_to_gpu (expecting non-exhausted dataloader iter)
        """

        # attach the model just in case the user forgets to call it, especially when the user
        # pauses the pipeline.progress and detach the model for other purpose.
        if not self._model_attached:
            self.attach(self._model)

        # fill the pipeline is only needed for the beginning when the pipeline (batches) is empty
        self.fill_pipeline(dataloader_iter)

        # here is the expected stop after exhausting all batches
        if not self.batches:
            raise StopIteration

        # TODO: Remove once Bulk Eval migrated (needed for bwd compat, this class only)
        self._set_module_context(self.contexts[0])

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        # wait for batches[0] being available on device, this should always be completed since
        # the input_dist of batches[0] has be invoked in previous iter. TODO: fact check
        self._wait_for_batch()

        if len(self.batches) >= 2:
            # invoke splits all_to_all comms (first part of input_dist)
            self.start_sparse_data_dist(self.batches[1], self.contexts[1])

        # batch i+2: load data and copy to gpu, the dataload iter will first exhaust here
        self.enqueue_batch(dataloader_iter)

        # forward
        with record_function("## forward ##"):
            losses, output = self._model_fwd(self.batches[0])

        if len(self.batches) >= 2:
            # invoke data (values, lengths, etc.) all_to_all comms (second part of input_dist)
            self.wait_sparse_data_dist(self.contexts[1])

        if self._model.training:
            # backward
            self._backward(losses)

            # update
            with record_function("## optimizer ##"):
                self._optimizer.step()

        self.dequeue_batch()
        return output

    def _create_context(self) -> TrainPipelineContext:
        context = self._context_type(index=self._next_index, version=1)
        self._next_index += 1
        return context

    def _pipeline_model(
        self,
        batch: Optional[In],
        context: TrainPipelineContext,
        pipelined_forward: Type[PipelinedForward] = PipelinedForward,
    ) -> None:
        (
            self._pipelined_modules,
            self._model,
            self._original_forwards,
            self._pipelined_postprocs,
            _,
        ) = _rewrite_model(
            model=self._model,
            context=context,
            dist_stream=self._data_dist_stream,
            default_stream=torch.get_device_module(self._device).current_stream(),
            batch=batch,
            apply_jit=self._apply_jit,
            pipelined_forward=pipelined_forward,
            pipeline_postproc=self._pipeline_postproc,
        )
        # initializes input dist, so we can override input dist forwards
        self.start_sparse_data_dist(batch, context)
        self._original_kjt_dist_forwards = _override_input_dist_forwards(
            self._pipelined_modules
        )

    def _init_pipelined_modules(
        self,
        batch: In,
        context: TrainPipelineContext,
        pipelined_forward: Type[PipelinedForward] = PipelinedForward,
    ) -> None:
        """
        Retrieves the pipelined modules after overriding their forwards, initializes the
        modules' input dists, and overrides the input dist forwards to support fusing
        the splits collective in the input dist.
        """
        if self._pipelined_modules:
            self._set_module_context(context)
            self.start_sparse_data_dist(batch, context)
            return

        self._pipeline_model(batch, context, pipelined_forward)

    def copy_batch_to_gpu(
        self,
        dataloader_iter: Iterator[In],
    ) -> Tuple[Optional[In], Optional[TrainPipelineContext]]:
        """
        Retrieves batch from dataloader and moves it to the provided device.

        Raises:
            StopIteration: if the dataloader iterator is exhausted; unless
                `self._execute_all_batches=True`, then returns None.
        """
        context = self._create_context()
        with record_function(f"## copy_batch_to_gpu {self._next_index} ##"):
            with self._stream_context(self._memcpy_stream):
                batch = self._next_batch(dataloader_iter)
                if batch is not None:
                    batch = _to_device(batch, self._device, non_blocking=True)
                elif not self._execute_all_batches:
                    raise StopIteration
                return batch, context

    def _next_batch(self, dataloader_iter: Iterator[In]) -> Optional[In]:
        """
        Retrieves next batch from dataloader and prevents calling `next` on an already
        exhausted dataloader, which can cause hanging.
        """
        if dataloader_iter is not self._dataloader_iter:
            self._dataloader_iter = dataloader_iter
            self._dataloader_exhausted = False

        if self._dataloader_exhausted:
            batch = None
        else:
            with record_function("## next_batch ##"):
                batch = next(dataloader_iter, None)
            if batch is None:
                self._dataloader_exhausted = True
        return batch

    def start_sparse_data_dist(
        self, batch: Optional[In], context: TrainPipelineContext
    ) -> None:
        """
        Waits for batch to finish getting copied to GPU, then starts the input dist.
        """
        if batch is None:
            return
        with record_function(f"## start_sparse_data_dist {context.index} ##"):
            with self._stream_context(self._data_dist_stream):
                _wait_for_batch(batch, self._memcpy_stream)

                original_contexts = [p.get_context() for p in self._pipelined_postprocs]

                # Temporarily set context for next iter to populate cache
                for postproc_mod in self._pipelined_postprocs:
                    postproc_mod.set_context(context)
                _start_data_dist(self._pipelined_modules, batch, context)

                # Restore context for model fwd
                for module, context in zip(
                    self._pipelined_postprocs, original_contexts
                ):
                    module.set_context(context)

    def wait_sparse_data_dist(self, context: TrainPipelineContext) -> None:
        """
        Waits on the input dist splits requests to get the input dist tensors requests,
        and populates the context with them.
        """
        with record_function(f"## wait_sparse_data_dist {context.index} ##"):
            with self._stream_context(self._data_dist_stream):
                for names, awaitable in context.fused_splits_awaitables:
                    for name, request in zip(names, awaitable.wait()):
                        context.input_dist_tensors_requests[name] = request
        context.input_dist_splits_requests.clear()
        context.fused_splits_awaitables.clear()

    def _copy_batch_to_gpu(self, dataloader_iter: Iterator[In]) -> Optional[In]:
        """
        DEPRECATED: exists for backward compatibility on TrainPipelineContext.version 0
        """
        self._set_module_context(self._context)
        batch, _ = self.copy_batch_to_gpu(dataloader_iter)
        return batch

    def _start_sparse_data_dist(self, batch: Optional[In]) -> None:
        """
        DEPRECATED: exists for backward compatibility
        Waits for batch to finish getting copied to GPU, then starts the input dist.
        """
        self._set_module_context(self._context)
        self.start_sparse_data_dist(batch, self._context)

    def _wait_sparse_data_dist(self) -> None:
        """
        DEPRECATED: exists for backward compatibility
        Waits on the input dist splits requests to get the input dist tensors requests,
        and populates the context with them.
        """
        self._set_module_context(self._context)
        with record_function("## wait_sparse_data_dist ##"):
            with self._stream_context(self._data_dist_stream):
                self._context.module_contexts = (
                    self._context.module_contexts_next_batch.copy()
                )
                self._context.input_dist_tensors_requests.clear()
                for names, awaitable in self._context.fused_splits_awaitables:
                    for name, request in zip(names, awaitable.wait()):
                        self._context.input_dist_tensors_requests[name] = request

    def _fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        """
        DEPRECATED: exists for backward compatibility
        """
        # pipeline is already filled
        if self._batch_i and self._batch_ip1:
            return
        # executes last batch in pipeline
        if self._batch_i and self._execute_all_batches:
            return

        # batch 1
        self._batch_i = self._copy_batch_to_gpu(dataloader_iter)
        if self._batch_i is None:
            raise StopIteration

        self._init_pipelined_modules(self._batch_i, self._context)
        self._start_sparse_data_dist(self._batch_i)
        self._wait_sparse_data_dist()

        # batch 2
        self._batch_ip1 = self._copy_batch_to_gpu(dataloader_iter)


class PrefetchTrainPipelineSparseDist(TrainPipelineSparseDist[In, Out]):
    """
    This pipeline overlaps device transfer, `ShardedModule.input_dist()`, and cache
    prefetching with forward and backward. This helps hide the all2all latency while
    preserving the training forward / backward ordering.

    stage 4: forward, backward - uses default CUDA stream
    stage 3: prefetch - uses prefetch CUDA stream
    stage 2: ShardedModule.input_dist() - uses data_dist CUDA stream
    stage 1: device transfer - uses memcpy CUDA stream

    `ShardedModule.input_dist()` is only done for top-level modules in the call graph.
    To be considered a top-level module, a module can only depend on 'getattr' calls on
    input.

    Input model must be symbolically traceable with the exception of `ShardedModule` and
    `DistributedDataParallel` modules.

    Args:
        model (torch.nn.Module): model to pipeline.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device where device transfer, sparse data dist, prefetch,
            and forward/backward pass will happen.
        execute_all_batches (bool): executes remaining batches in pipeline after
            exhausting dataloader iterator.
        apply_jit (bool): apply torch.jit.script to non-pipelined (unsharded) modules.
    """

    # The PipelinedForward class that is used in _rewrite_model
    _pipelined_forward_type = PrefetchPipelinedForward  # pyre-ignore

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        pipeline_postproc: bool = True,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            execute_all_batches=execute_all_batches,
            apply_jit=apply_jit,
            context_type=PrefetchTrainPipelineContext,
            pipeline_postproc=pipeline_postproc,
            custom_model_fwd=custom_model_fwd,
        )
        self._context = PrefetchTrainPipelineContext(version=0)
        self._prefetch_stream: Optional[torch.Stream] = (
            (torch.get_device_module(device).Stream())
            if self._device.type in ["cuda", "mtia"]
            else None
        )
        self._default_stream: Optional[torch.Stream] = (
            (torch.get_device_module(self._device).Stream())
            if self._device.type in ["cuda", "mtia"]
            else None
        )
        self._batch_ip3: Optional[In] = None

    def _fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        # pipeline is already filled
        if self._batch_i and self._batch_ip1 and self._batch_ip2:
            return
        # executes last batch in pipeline
        if self._execute_all_batches and (self._batch_i or self._batch_ip1):
            return

        # batch 1
        self._batch_i = self._copy_batch_to_gpu(dataloader_iter)
        if self._batch_i is None:
            raise StopIteration

        self._init_pipelined_modules(
            self._batch_i,
            self._context,
            # pyre-ignore
            self._pipelined_forward_type,
        )
        self._start_sparse_data_dist(self._batch_i)
        self._wait_sparse_data_dist()
        self._prefetch(self._batch_i)

        # batch 2
        self._batch_ip1 = self._copy_batch_to_gpu(dataloader_iter)
        self._start_sparse_data_dist(self._batch_ip1)

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        self._fill_pipeline(dataloader_iter)

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self._batch_i), self._prefetch_stream)

        self._batch_ip2 = self._copy_batch_to_gpu(dataloader_iter)

        self._wait_sparse_data_dist()
        # forward
        with record_function("## forward ##"):
            losses, output = self._model_fwd(self._batch_i)

        self._prefetch(self._batch_ip1)

        if self._model.training:
            # backward
            with record_function("## backward ##"):
                torch.sum(losses, dim=0).backward()

            # update
            with record_function("## optimizer ##"):
                self._optimizer.step()

        self._start_sparse_data_dist(self._batch_ip2)

        self._batch_i = self._batch_ip1
        self._batch_ip1 = self._batch_ip2

        return output

    def _prefetch(self, batch: Optional[In]) -> None:
        """
        Waits for input dist to finish, then prefetches data.
        """
        if batch is None:
            return
        self._context.module_input_post_prefetch.clear()
        self._context.module_contexts_post_prefetch.clear()

        with record_function("## sharded_module_prefetch ##"):
            with self._stream_context(self._prefetch_stream):
                batch.record_stream(
                    torch.get_device_module(self._device).current_stream()
                )
                data_per_pipelined_module = _prefetch_embeddings(
                    batch,
                    self._context,
                    self._pipelined_modules,
                    self._device,
                    self._stream_context,
                    self._data_dist_stream,
                    self._default_stream,
                )
                for sharded_module in self._pipelined_modules:
                    forward = sharded_module.forward
                    data = data_per_pipelined_module[forward._name]
                    self._context.module_input_post_prefetch[forward._name] = data
                    self._context.module_contexts_post_prefetch[
                        forward._name
                    ] = self._context.module_contexts.pop(forward._name)


class JaggedMegatronTrainPipelineSparseDist(TrainPipelineSparseDist[In, Out]):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        # keep for backward compatibility
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            device,
            execute_all_batches,
            apply_jit,
            TrainPipelineContext,
            pipeline_postproc,
            custom_model_fwd,
        )

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        """
        For TrainPipelineSparseDist, we assume the max pipelined batches == 3 (capacity):
            batches[0]: current batch, for emb_lookup, output_dist, and fwd/bwd/opt (expecting input_dist)
            batches[1]: next batch, for input_dist (expecting copied to device)
            batches[2]: i+2 batch, for copy_batch_to_gpu (expecting non-exhausted dataloader iter)
        """

        # attach the model just in case the user forgets to call it, especially when the user
        # pauses the pipeline.progress and detach the model for other purpose.
        if not self._model_attached:
            self.attach(self._model)

        # fill the pipeline is only needed for the beginning when the pipeline (batches) is empty
        self.fill_pipeline(dataloader_iter)

        # here is the expected stop after exhausting all batches
        if not self.batches:
            raise StopIteration

        # TODO: Remove once Bulk Eval migrated (needed for bwd compat, this class only)
        self._set_module_context(self.contexts[0])

        if self._model.training:
            with nvtx.annotate("## zero_grad ##"):
                if hasattr(self._model.module, "zero_grad_buffer"):
                    self._model.module.zero_grad_buffer()
                self._optimizer.zero_grad()

        # wait for batches[0] being available on device, this should always be completed since
        # the input_dist of batches[0] has be invoked in previous iter. TODO: fact check
        with nvtx.annotate("## wait_for_batch ##"):
            self._wait_for_batch()

        with nvtx.annotate("## start_sparse_data_dist ##"):
            if len(self.batches) >= 2:
                # invoke splits all_to_all comms (first part of input_dist)
                self.start_sparse_data_dist(self.batches[1], self.contexts[1])

        # batch i+2: load data and copy to gpu, the dataload iter will first exhaust here
        with nvtx.annotate("## enqueue_batch ##"):
            self.enqueue_batch(dataloader_iter)

        # forward
        with nvtx.annotate("## forward ##"):
            losses, output = self._model_fwd(self.batches[0])
        with nvtx.annotate("## loss postprocess ##"):
            collective_assert(not torch.isnan(losses).any(), "loss has nan value")
            local_tokens = torch.tensor(losses.size(0), device=self._device).float()
            local_loss = torch.cat([torch.sum(losses).view(1), local_tokens.view(1)])
            reporting_loss = local_loss.clone().detach()
            torch.distributed.all_reduce(
                reporting_loss, group=parallel_state.get_data_parallel_group()
            )
        if len(self.batches) >= 2:
            # invoke data (values, lengths, etc.) all_to_all comms (second part of input_dist)
            self.wait_sparse_data_dist(self.contexts[1])

        if self._model.training:
            # backward
            with nvtx.annotate("## backward ##"):
                dp_size = parallel_state.get_data_parallel_world_size()
                # in case of uneven jagged size across dp ranks.
                local_loss_average = local_loss[0] / reporting_loss[1] * dp_size
                local_loss_average.backward()

            with nvtx.annotate("## finalize_model_grads ##"):
                if isinstance(self._model.module, DistributedDataParallel):
                    finalize_model_grads([self._model.module], None)
            # update
            with nvtx.annotate("## optimizer ##"):
                self._optimizer.step()
        self.dequeue_batch()
        return reporting_loss, output


class JaggedMegatronPrefetchTrainPipelineSparseDist(
    PrefetchTrainPipelineSparseDist[In, Out]
):
    def __init__(
        self,
        model: torch.nn.Module,  # might be wrapped by DistributedModelParallel
        optimizer: torch.optim.Optimizer,  # dense optimizer, might be a megatron optimizer
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        pipeline_postproc: bool = True,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            device,
            execute_all_batches,
            apply_jit,
            pipeline_postproc,
            custom_model_fwd,
        )

    def progress(self, dataloader_iter: Iterator[In]) -> Tuple[torch.Tensor, Out]:
        self._fill_pipeline(dataloader_iter)

        if self._model.training:
            with nvtx.annotate("## zero_grad ##"):
                if hasattr(self._model.module, "zero_grad_buffer"):
                    self._model.module.zero_grad_buffer()
                self._optimizer.zero_grad()

        with nvtx.annotate("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self._batch_i), self._prefetch_stream)

        with nvtx.annotate("## copy_batch_to_gpu ##"):
            self._batch_ip2 = self._copy_batch_to_gpu(dataloader_iter)
        with nvtx.annotate("## wait_sparse_data_dist ##"):
            self._wait_sparse_data_dist()
        # forward
        reporting_loss = None
        with nvtx.annotate("## forward ##"):
            losses, output = self._model_fwd(self._batch_i)
        with nvtx.annotate("## loss postprocess ##"):
            collective_assert(not torch.isnan(losses).any(), "loss has nan value")
            local_tokens = torch.tensor(losses.size(0), device=self._device).float()
            local_loss = torch.cat([torch.sum(losses).view(1), local_tokens.view(1)])
            reporting_loss = local_loss.clone().detach()
            torch.distributed.all_reduce(
                reporting_loss, group=parallel_state.get_data_parallel_group()
            )
        with nvtx.annotate("## prefetch ##"):
            self._prefetch(self._batch_ip1)

        if self._model.training:
            # backward
            with nvtx.annotate("## backward ##"):
                dp_size = parallel_state.get_data_parallel_world_size()
                # in case of uneven jagged size across dp ranks.
                # this cannot solve the issue completely because of the allreduce avg across DP ranks. To solve it completely, we need to perform sum reduce.
                # see https://github.com/NVIDIA/Megatron-LM/blob/v0.12.0rc3/megatron/core/distributed/distributed_data_parallel.py#L237-L240
                local_loss_average = local_loss[0] / reporting_loss[1] * dp_size
                local_loss_average.backward()

                # self._model is a DistributedModelParallel
                # self._model.module could be is a DistributedDataParallel
                if isinstance(self._model.module, DistributedDataParallel):
                    finalize_model_grads([self._model.module], None)

            # update
            with nvtx.annotate("## optimizer ##"):
                self._optimizer.step()

        with nvtx.annotate("## input_dist ##"):
            self._start_sparse_data_dist(self._batch_ip2)

        self._batch_i = self._batch_ip1
        self._batch_ip1 = self._batch_ip2

        return reporting_loss, output


class JaggedMegatronTrainNonePipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        self._model = model
        self._optimizer = optimizer
        self._device = device

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        dp_size = parallel_state.get_data_parallel_world_size() * 1.0
        with nvtx.annotate("## zero_grad ##"):
            if hasattr(self._model.module, "zero_grad_buffer"):
                self._model.module.zero_grad_buffer()
            self._optimizer.zero_grad()
        with nvtx.annotate("## H2D ##"):
            batch = next(dataloader_iter).to(self._device)
            # print(f'nopipeline batch.features: {batch.features.values()}')

        with nvtx.annotate("## forward ##"):
            losses, output = self._model(batch)

        with nvtx.annotate("## loss postprocess ##"):
            collective_assert(not torch.isnan(losses).any(), "loss has nan value")
            local_tokens = torch.tensor(
                losses.size(0), device=torch.device("cuda", torch.cuda.current_device())
            ).float()
            local_loss = torch.cat([torch.sum(losses).view(1), local_tokens.view(1)])
            reporting_loss = local_loss.clone().detach()
            # [allreduced_sum_loss, allreduced_sum_tokens]
            torch.distributed.all_reduce(
                reporting_loss, group=parallel_state.get_data_parallel_group()
            )
        if self._model.training:
            with nvtx.annotate("## backward ##"):
                local_loss_average = local_loss[0] / reporting_loss[1] * dp_size
                local_loss_average.backward()

            with nvtx.annotate("## finalize_model_grads ##"):
                if isinstance(self._model.module, DistributedDataParallel):
                    finalize_model_grads([self._model.module], None)

            with nvtx.annotate("## optimizer step ##"):
                self._optimizer.step()

        return reporting_loss, output
