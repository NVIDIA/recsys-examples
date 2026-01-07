from typing import List, Optional, Tuple, Union

import torch


# TODO, make it graphable
class BeamSearch:
    def __init__(
        self,
        beam_width: Union[int, List[int]],
        num_hierarchies: int,
        codebook_sizes: List[int],
        codebooks: Optional[
            torch.Tensor
        ] = None,  # to check if the sid is mapped into the codebook
        prefix_valid_check: bool = False,  # to check if the prefix is valid
        record_history: bool = False,
    ):
        """
        codebooks : [num_items, num_hierarchies]
        """
        if isinstance(beam_width, int):
            beam_widths = [beam_width] * num_hierarchies
        self.beam_widths = beam_widths
        self.num_hierarchies = num_hierarchies
        self.codebook_sizes = codebook_sizes
        assert (
            len(codebook_sizes) == num_hierarchies
        ), "codebook_sizes should be the same length as num_hierarchies"

        if prefix_valid_check:
            assert (
                codebooks is not None
            ), "codebooks should be provided if prefix_valid_check is True"
        self.accumulated_log_probs: torch.Tensor = torch.tensor(
            []
        )  # to perceive the mppy check
        self.generated_sids: torch.Tensor = torch.tensor(
            []
        )  # to perceive the mppy check
        self.step: int = 0

        # for debugging purpose
        self.record_history: bool = record_history
        self.history_topk_sids: List[torch.Tensor] = []
        self.history_accumulate_topk_probs: List[torch.Tensor] = []
        self.history_probs: List[torch.Tensor] = []
        self.reset()

    def propagate(
        self,
        log_probs: torch.Tensor,  # [batch_size, topk_previous_step, codebook_size[step]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        In the beginning of step i, we have already generated sids [batchsize, beam_widths[i-1], i],
        We will extend the generated sids [batchsize, beam_widths[i], i + 1] with the log_probs [batchsize, codebook_size[i]]
        """
        step = self.step
        if step >= self.num_hierarchies:
            raise ValueError(
                "Reached the last hierarchy, please call reset to start a new generation"
            )
        batch_size, codebook_size_this_step = log_probs.shape[0], log_probs.shape[-1]
        topk_previous_step = self.generated_sids.shape[1] if step > 0 else 1
        topk_this_step = min(
            self.beam_widths[step], topk_previous_step * codebook_size_this_step
        )

        if step == 0:
            # initialize the generated sids and accumulated log probs
            self.generated_sids = torch.empty(
                batch_size,
                topk_previous_step,
                step,
                device=log_probs.device,
                dtype=torch.long,
            )
            self.accumulated_log_probs = torch.zeros(
                batch_size,
                topk_previous_step,
                device=log_probs.device,
                dtype=torch.float,
            )

        log_probs_this_step = log_probs.view(
            batch_size, topk_previous_step, codebook_size_this_step
        )
        accumulated_log_probs_this_step = (
            self.accumulated_log_probs.view(batch_size, topk_previous_step, 1)
            + log_probs_this_step
        )
        # [batch_size, topk_previous_step * codebook_size_this_step]
        accumulated_log_probs_this_step = accumulated_log_probs_this_step.view(
            batch_size, -1
        )
        topk_probs, topk_indices = torch.topk(
            accumulated_log_probs_this_step, topk_this_step, dim=-1
        )
        current_step_sids = topk_indices % codebook_size_this_step
        last_step_indices = topk_indices // codebook_size_this_step
        # [batch_size, topk_this_step, step]
        # it's safe to expand to zero when step is 0,
        last_step_indices_expanded = last_step_indices.unsqueeze(-1).expand(
            -1, -1, step
        )
        last_step_sids = torch.gather(
            self.generated_sids, dim=1, index=last_step_indices_expanded
        )
        generated_sids = torch.cat(
            [last_step_sids, current_step_sids.unsqueeze(-1)], dim=-1
        )
        if self.record_history:
            self.history_topk_sids.append(generated_sids)
            self.history_accumulate_topk_probs.append(torch.exp(topk_probs))
            self.history_probs.append(torch.exp(log_probs_this_step))
        self.generated_sids = generated_sids
        self.accumulated_log_probs = topk_probs
        self.step += 1
        # [[maybe discard]]
        return generated_sids, topk_probs

    def reset(self):
        self.generated_sids = None
        self.accumulated_log_probs = None
        self.step = 0
        self.history_topk_sids = []
        self.history_accumulate_topk_probs = []
        self.history_probs = []

    def get_sids(
        self,
        step: Optional[int] = None,  # [-1 ~ num_hierarchies)
    ) -> torch.Tensor:
        """
        return the generated sids at step i if step is valid, otherwise return None.
        """
        if step is None:
            return self.generated_sids
        elif step == -1:
            return None
        elif step < self.step:
            return self.generated_sids[:, :, step]
        else:
            raise ValueError(f"Step {step} is not valid, current step is {self.step}")

    def generate_valid_mask(self) -> torch.Tensor:
        """
        update the valid mask between current step and previous step,
        this can be used for transformer attention
        """

    def get_log_probs(self) -> torch.Tensor:
        return self.accumulated_log_probs
