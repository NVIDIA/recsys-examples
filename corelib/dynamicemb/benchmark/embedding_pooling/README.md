## Result
### Correctness
```
================================================================================
Correctness Testing
================================================================================

Small segments:
  batch=100, dim=128, avg_len=10
  sum : ref=1.91e-06, torch=2.86e-06, cuda=2.86e-06 ✓ ✓
  mean: ref=2.38e-07, torch=2.38e-07, cuda=2.38e-07 ✓ ✓

Medium segments:
  batch=1000, dim=256, avg_len=50
  sum : ref=7.63e-06, torch=9.54e-06, cuda=1.14e-05 ✓ ✓
  mean: ref=1.19e-07, torch=1.79e-07, cuda=2.09e-07 ✓ ✓

Large segments:
  batch=500, dim=512, avg_len=100
  sum : ref=9.54e-06, torch=1.91e-05, cuda=2.29e-05 ✓ ✓
  mean: ref=8.94e-08, torch=2.09e-07, cuda=2.38e-07 ✓ ✓

Many segments:
  batch=10000, dim=128, avg_len=20
  sum : ref=7.63e-06, torch=5.72e-06, cuda=5.72e-06 ✓ ✓
  mean: ref=3.58e-07, torch=2.38e-07, cuda=2.38e-07 ✓ ✓

Mixed lengths:
  batch=1000, dim=128, avg_len=None
  sum : ref=5.72e-06, torch=1.53e-05, cuda=1.53e-05 ✓ ✓
  mean: ref=2.38e-07, torch=2.38e-07, cuda=2.38e-07 ✓ ✓

Edge cases (with empty segments):
  Triton: diff=1.19e-07 ✓
  CUDA:   diff=1.19e-07 ✓
```

### Benchmark
```
================================================================================
Performance Benchmarking
================================================================================

Small segs: 1000 segs, dim=128, avg_len=10, total=9699
  → Strategy: Triton

  Results:
    Triton:        0.1045 ms
    CUDA kernel:   0.0173 ms  (vs Triton:  6.03x)
    PyTorch:       0.2498 ms  (vs Triton:  2.39x)
    Reference:    55.2461 ms  (speedup: 528.6x)

Medium segs: 1000 segs, dim=256, avg_len=50, total=49637
  → Strategy: Triton

  Results:
    Triton:        0.0627 ms
    CUDA kernel:   0.0590 ms  (vs Triton:  1.06x)
    PyTorch:       0.3299 ms  (vs Triton:  5.26x)
    Reference:    57.8561 ms  (speedup: 923.1x)

Large segs: 500 segs, dim=256, avg_len=150, total=74799
  → Strategy: Triton

  Results:
    Triton:        0.0885 ms
    CUDA kernel:   0.0751 ms  (vs Triton:  1.18x)
    PyTorch:       0.4140 ms  (vs Triton:  4.68x)
    Reference:    28.9529 ms  (speedup: 327.0x)

Very large: 100 segs, dim=512, avg_len=600, total=59940
  → Strategy: Triton

  Results:
    Triton:        0.2723 ms
    CUDA kernel:   0.2243 ms  (vs Triton:  1.21x)
    PyTorch:       0.5808 ms  (vs Triton:  2.13x)
    Reference:     8.9676 ms  (speedup:  32.9x)

Many small: 10000 segs, dim=128, avg_len=20, total=195248
  → Strategy: PyTorch

  Results:
    Triton:        0.5239 ms
    CUDA kernel:   0.0977 ms  (vs Triton:  5.36x)
    PyTorch:       0.5236 ms  (vs Triton:  1.00x)
    Reference:    491.1181 ms  (speedup: 937.5x)

Huge batch: 20000 segs, dim=128, avg_len=15, total=290518
  → Strategy: PyTorch

  Results:
    Triton:        0.7288 ms
    CUDA kernel:   0.1490 ms  (vs Triton:  4.89x)
    PyTorch:       0.7325 ms  (vs Triton:  1.01x)

================================================================================
✓ All tests completed
================================================================================
```