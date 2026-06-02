"""Runtime contracts for GR inference."""

from gr_inference.gr_runtime.beam_search import (
    BeamSelection,
    InitialTopKBeamSelector,
    item_mask_limited_beam_width,
    select_initial_topk,
    select_next_topk,
)
from gr_inference.gr_runtime.batched_beam_search import (
    BatchedBeamSelection,
    batched_item_mask_limited_beam_width,
    select_initial_topk_batched,
    select_next_topk_batched,
)
from gr_inference.gr_runtime.batched_decode_inputs import (
    BatchedDecodeInputs,
    make_batched_beam_token_ids,
)
from gr_inference.gr_runtime.batched_topk_indices import (
    make_batched_topk_indices,
    make_compacted_batched_topk_indices,
)
from gr_inference.gr_runtime.beam_kv_compaction import (
    compact_batched_beam_kv_history,
    needs_batched_beam_kv_history_compaction,
)
from gr_inference.gr_runtime.decode_kv import BeamKVWrite, BeamKVWriter
from gr_inference.gr_runtime.decode_loop import (
    DecodeLoopResult,
    DecodeLoopStep,
    FixedBeamDecodeLoop,
)
from gr_inference.gr_runtime.engine import GRDecodeEngine
from gr_inference.gr_runtime.generation import (
    GRGenerationState,
    PrefillResult,
    allocate_beam_kv_like_context,
)
from gr_inference.gr_runtime.item_constraints import (
    SemanticItemCatalog,
    SemanticItem,
    TokenTrie,
    TrieItemMaskProvider,
    TrieItemMaskProviderStore,
)
from gr_inference.gr_runtime.logits_processor import (
    LogitsProcessorContext,
    TokenBiasLogitsProcessor,
    TokenSuppressLogitsProcessor,
    apply_logits_processors,
    logits_processor_from_spec,
    logits_processors_from_specs,
    logits_processors_metadata,
    validate_logits_processors,
)
from gr_inference.gr_runtime.outputs import DecodeStepOutput
from gr_inference.gr_runtime.prefill import ContextKVWriter, GRPrefillRunner
from gr_inference.gr_runtime.request import GRRequestState
from gr_inference.gr_runtime.timing import TimingRecorder

__all__ = [
    "BeamSelection",
    "BatchedBeamSelection",
    "BatchedDecodeInputs",
    "BeamKVWrite",
    "BeamKVWriter",
    "ContextKVWriter",
    "DecodeStepOutput",
    "DecodeLoopResult",
    "DecodeLoopStep",
    "FixedBeamDecodeLoop",
    "GRDecodeEngine",
    "GRGenerationState",
    "GRPrefillRunner",
    "InitialTopKBeamSelector",
    "LogitsProcessorContext",
    "PrefillResult",
    "GRRequestState",
    "SemanticItem",
    "SemanticItemCatalog",
    "TokenBiasLogitsProcessor",
    "TokenSuppressLogitsProcessor",
    "TokenTrie",
    "TrieItemMaskProvider",
    "TrieItemMaskProviderStore",
    "TimingRecorder",
    "allocate_beam_kv_like_context",
    "apply_logits_processors",
    "batched_item_mask_limited_beam_width",
    "item_mask_limited_beam_width",
    "logits_processor_from_spec",
    "logits_processors_from_specs",
    "logits_processors_metadata",
    "select_initial_topk",
    "select_initial_topk_batched",
    "select_next_topk_batched",
    "make_batched_beam_token_ids",
    "make_batched_topk_indices",
    "make_compacted_batched_topk_indices",
    "compact_batched_beam_kv_history",
    "needs_batched_beam_kv_history_compaction",
    "select_next_topk",
    "validate_logits_processors",
]
