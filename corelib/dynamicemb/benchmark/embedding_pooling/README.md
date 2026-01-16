## Description
A embedding pooling kernel to address performance issues with `DynamicEmbeddingBagFunction`.

### Core Implementation
- `embedding_pooling_kernel.py`: Triton kernels (forward + backward)
- `embedding_pooling.py`: PyTorch autograd integration
- `test_embedding_pooling.py`: Unified test suite

## Important: Triton Version Requirement

When using Docker images built from the 25.10 release or earlier Dockerfiles, this kernel may produce numerical errors due to an outdated Triton version. To fix this issue, please manually upgrade Triton:

```bash
pip install triton==3.3.0
```

## Usage Example

```python
from embedding_pooling import embedding_pooling

# Get embeddings from DynamicEmbeddingFunctionV2
embeddings = DynamicEmbeddingFunctionV2(...)  # [total_embs, dim]

# Apply pooling
pooled = embedding_pooling(embeddings, offsets, "mean")  # [batch, dim]
```
## Performance

| Test Case       | Config (Batch/Dim/Len) | Fwd-Triton | Fwd-PyTorch | Bwd-Triton | Bwd-PyTorch |
|:----------------|:--------------------|------------|-------------|------------|-------------|
| Small segments  | 100 / 128 / 10      | 2.752 μs | 21.76 μs | 3.2 μs | 7.808 μs |
| Medium segments | 1000 / 256 / 50     | 43.073 μs | 75.873 μs | 103.745 μs | 134.625 μs |
| Large segments  | 500 / 512 / 100     | 71.936 μs | 128.160 μs | 213.314 μs | 268.451 μs |
| Many segments   | 10000 / 128 / 20    | 70.144 μs  | 135.458 μs | 216.353 μs | 261.026 μs |
| Mixed lengths   | 1000 / 128 / None   | 28.352 μs | 52.195 μs | 47.681 μs | 63.073 μs |

