from typing import Callable, Iterator, Optional, Tuple, TypeVar, cast

import torch
from commons.utils.distributed_utils import collective_assert
from megatron.core import parallel_state
from megatron.core.distributed import finalize_model_grads
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from torch.autograd.profiler import record_function
from torchrec.distributed.train_pipeline import PrefetchTrainPipelineSparseDist
from torchrec.distributed.train_pipeline.utils import _wait_for_batch
from torchrec.streamable import Pipelineable

In = TypeVar("In", bound=Pipelineable)
Out = TypeVar("Out")


class JaggedMegatronPrefetchTrainPipelineSparseDist(PrefetchTrainPipelineSparseDist):
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
        self._batch_i: In
        self._batch_ip1: In
        self._batch_ip2: In
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
            with record_function("## zero_grad ##"):
                if hasattr(self._model.module, "zero_grad_buffer"):
                    self._model.module.zero_grad_buffer()
                self._optimizer.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self._batch_i), self._prefetch_stream)

        self._batch_ip2 = self._copy_batch_to_gpu(dataloader_iter)

        self._wait_sparse_data_dist()
        # forward
        reporting_loss = None
        with record_function("## forward ##"):
            losses, output = self._model_fwd(self._batch_i)
            collective_assert(not torch.isnan(losses).any(), "loss has nan value")
            local_tokens = torch.tensor(losses.size(0), device=self._device).float()
            local_loss = torch.cat([torch.sum(losses).view(1), local_tokens.view(1)])
            reporting_loss = local_loss.clone().detach()
        self._prefetch(self._batch_ip1)

        if self._model.training:
            # backward
            with record_function("## backward ##"):
                dp_size = parallel_state.get_data_parallel_world_size()
                local_loss_average = local_loss[0] / reporting_loss[1] * dp_size
                local_loss_average.backward()
                torch.distributed.all_reduce(
                    reporting_loss, group=parallel_state.get_data_parallel_group()
                )
                # self._model is a DistributedModelParallel
                # self._model.module could be is a DistributedDataParallel
                if isinstance(self._model.module, DistributedDataParallel):
                    finalize_model_grads([self._model.module], None)

            # update
            with record_function("## optimizer ##"):
                self._optimizer.step()

        self._start_sparse_data_dist(self._batch_ip2)

        self._batch_i = self._batch_ip1
        self._batch_ip1 = self._batch_ip2

        return reporting_loss, output
