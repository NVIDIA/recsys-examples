# SID-GR Training Example

This example implements a retrieval model with a standard Transformer decoder backbone. We use [Gin config](https://github.com/google/gin-config) to specify model hyperparameters and training configurations (e.g., dataset file paths, training steps). Currently, this implementation has been validated on the Amazon Beauty dataset.

For detailed information about the Gin config interface and available parameters, please refer to the [inline documentation](../configs/sid_gin_config_args.py).

## Important: `max_candidate_length` Configuration

The `DatasetArgs.max_candidate_length` parameter controls which items in the sequence are used for loss calculation:

- `max_candidate_length=1`: Only the **last item** in the sequence is used to calculate loss
- `max_candidate_length=0`: **All items except the first one** in the sequence are used to calculate loss

## Dataset Preprocessing

This example requires two types of dataset files:

1. **PID-to-SID mapping tensor**: A PyTorch tensor file (loadable via `torch.load()`)
2. **Historical interaction sequences**: Parquet format files containing user-item interaction histories

### PID-to-SID Mapping

The PID-to-SID tokenization process (i.e., converting product IDs to semantic IDs) is **not included** in this example. Users must tokenize items separately before training. We recommend using [GRID](https://github.com/snap-research/GRID) for this purpose. 

**Requirements:**
- The mapping tensor must have shape: `[num_hierarchies, num_items]`
- The tensor should be compatible with `torch.load()`

### Historical Sequence File

Similar to other sequential models (e.g., HSTU), each user has a historical interaction sequence with items. This example uses the **Parquet format**, which offers superior file compression and I/O performance.

**File structure:**
- Each row represents a user's interaction history
- The nested column contains the sequential history (variable length supported)
- Optional columns: user ID, sequence length
- A single user may span multiple rows with varying sequence lengths

### Dataset Statistics

| Dataset       | # Users | Max Seq Len | Min Seq Len | Mean Seq Len | Median Seq Len | # Items |
|---------------|---------|-------------|-------------|--------------|----------------|---------|
| Amazon Beauty | 22,363  | 202         | 3           | 7            | 4              | 12,101  |

## Jagged Tensor Support

This implementation assumes variable-length (jagged) input sequences. We leverage [TorchRec Jagged Tensor](https://docs.pytorch.org/tutorials/intermediate/torchrec_intro_tutorial.html#torchrec-input-output-data-types) utilities to efficiently handle jagged tensor operations. 

**Note:** Jagged tensors are also referred to as the `THD` (Total, Head, Dim) layout in [Megatron-Core / Transformer-Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/c/fused_attn.html#_CPPv4N15NVTE_QKV_Format8NVTE_THDE). 

## Getting Started

### 1. Download Dataset

Download the demo dataset from [Hugging Face](https://huggingface.co/datasets/DuDoddle/sid-amazon-beauty/) and ensure the data paths are correctly configured in the [Gin config file](../configs/sid_amazn.gin).

### 2. Run Training

The training entry point is [pretrain_sid_gr.py](./pretrain_sid_gr.py).

**Command to train on Amazon Beauty dataset:**

```bash
# Navigate to the sid_gr directory
cd <path-to-project>/examples/sid_gr

# Run training with 1 GPU
PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun \
  --nproc_per_node 1 \
  --master_addr localhost \
  --master_port 6000 \
  ./training/pretrain_sid_gr.py \
  --gin-config-file ./configs/sid_amazn.gin
```

**Note:** Ensure your current working directory is `examples/sid_gr` before running the command.

## Known Limitations

⚠️ **This implementation is under active development.** The current version has not been fully optimized for performance. Known limitations include:

- **Attention mechanism**: Currently using padded local SDPA (Scaled Dot-Product Attention) implementation in Megatron-Core with explicit attention masks
- **Beam search**: The beam search used during evaluation does not yet support KV cache optimization
- **Performance**: The model performance has not reached optimal levels

We are actively working on addressing these limitations and improving overall efficiency.