## Backward result
### Benchmark(cuda event)
```
================================================================================
Forward Performance Benchmarking
================================================================================

Small segs: 1000 segs, dim=128, avg_len=10, total=9699
  → Strategy: Triton

  Results:
    Triton:        0.0613 ms
    CUDA kernel:   0.0088 ms  (vs Triton:  6.98x)
    PyTorch:       0.1925 ms  (vs Triton:  3.14x)
    Reference:    52.3397 ms  (speedup: 853.5x)

Medium segs: 1000 segs, dim=256, avg_len=50, total=49637
  → Strategy: Triton

  Results:
    Triton:        0.0484 ms
    CUDA kernel:   0.0280 ms  (vs Triton:  1.73x)
    PyTorch:       0.1294 ms  (vs Triton:  2.68x)
    Reference:    46.3950 ms  (speedup: 959.4x)

Large segs: 500 segs, dim=256, avg_len=150, total=74799
  → Strategy: Triton

  Results:
    Triton:        0.0404 ms
    CUDA kernel:   0.0391 ms  (vs Triton:  1.03x)
    PyTorch:       0.1480 ms  (vs Triton:  3.67x)
    Reference:    23.1728 ms  (speedup: 573.8x)
Very large: 100 segs, dim=512, avg_len=600, total=59940
  → Strategy: Triton

  Results:
    Triton:        0.0524 ms
    CUDA kernel:   0.1097 ms  (vs Triton:  0.48x)
    PyTorch:       0.2631 ms  (vs Triton:  5.02x)
    Reference:     5.4698 ms  (speedup: 104.4x)

Many small: 10000 segs, dim=128, avg_len=20, total=195248
  → Strategy: PyTorch

  Results:
    Triton:        0.0456 ms
    CUDA kernel:   0.0451 ms  (vs Triton:  1.01x)
    PyTorch:       0.1713 ms  (vs Triton:  3.76x)
    Reference:    461.8171 ms  (speedup: 10137.6x)

Huge batch: 20000 segs, dim=128, avg_len=15, total=290518
  → Strategy: PyTorch

  Results:
    Triton:        0.0589 ms
    CUDA kernel:   0.0667 ms  (vs Triton:  0.88x)
    PyTorch:       0.2192 ms  (vs Triton:  3.72x)

================================================================================
Backward Performance Benchmarking
================================================================================

Small segs: 1000 segs, dim=128, avg_len=10, total=10096
  Results:
    Triton:    0.0824 ms
    PyTorch:   0.1888 ms  (ratio:  2.29x)
    Diff: 1.19e-07 ✓

Medium segs: 1000 segs, dim=256, avg_len=50, total=49767
  Results:
    Triton:    0.0936 ms
    PyTorch:   0.1969 ms  (ratio:  2.10x)
    Diff: 7.45e-09 ✓

Large segs: 500 segs, dim=256, avg_len=150, total=74505
  Results:
    Triton:    0.0965 ms
    PyTorch:   0.2413 ms  (ratio:  2.50x)
    Diff: 1.86e-09 ✓

Very large: 100 segs, dim=512, avg_len=600, total=59989
  Results:
    Triton:    0.1137 ms
    PyTorch:   0.3314 ms  (ratio:  2.92x)
    Diff: 4.66e-10 ✓

Many small: 10000 segs, dim=128, avg_len=20, total=195425
  Results:
    Triton:    0.1100 ms
    PyTorch:   0.2832 ms  (ratio:  2.57x)
    Diff: 2.98e-08 ✓

================================================================================
Complete Forward + Backward Benchmarking
================================================================================

Medium: 1000 segs, dim=256, total=49514
  Forward:   0.1678 ms
  Backward:  1.4103 ms
  Total:     1.5781 ms
  Backward/Forward ratio: 8.41x

Large: 500 segs, dim=512, total=49838
  Forward:   0.1704 ms
  Backward:  1.3918 ms
  Total:     1.5622 ms
  Backward/Forward ratio: 8.17x

Many segments: 10000 segs, dim=128, total=194798
  Forward:   0.1708 ms
  Backward:  1.3953 ms
  Total:     1.5661 ms
  Backward/Forward ratio: 8.17x

Mixed lengths: 1000 segs, dim=128, total=49352
  Forward:   0.1572 ms
  Backward:  1.3798 ms
  Total:     1.5370 ms
  Backward/Forward ratio: 8.78x
```

### Benchmark(forward+backward)
```
================================================================================
Forward Performance Benchmarking
================================================================================

Small segs: 1000 segs, dim=128, avg_len=10, total=9699
  → Strategy: Triton

  Results:
    Triton:        0.0575 ms
    CUDA kernel:   0.0094 ms  (vs Triton:  6.11x)
    PyTorch:       0.1974 ms  (vs Triton:  3.43x)

Medium segs: 1000 segs, dim=256, avg_len=50, total=49404
  → Strategy: Triton

  Results:
    Triton:        0.0570 ms
    CUDA kernel:   0.0283 ms  (vs Triton:  2.01x)
    PyTorch:       0.1981 ms  (vs Triton:  3.48x)

Large segs: 500 segs, dim=256, avg_len=150, total=74863
  → Strategy: Triton

  Results:
    Triton:        0.0569 ms
    CUDA kernel:   0.0394 ms  (vs Triton:  1.44x)
    PyTorch:       0.1978 ms  (vs Triton:  3.48x)

Very large: 100 segs, dim=512, avg_len=600, total=59969
  → Strategy: Triton

  Results:
    Triton:        0.0578 ms
    CUDA kernel:   0.1103 ms  (vs Triton:  0.52x)
    PyTorch:       0.2976 ms  (vs Triton:  5.15x)
Many small: 10000 segs, dim=128, avg_len=20, total=195121
  → Strategy: PyTorch

  Results:
    Triton:        0.0573 ms
    CUDA kernel:   0.0456 ms  (vs Triton:  1.26x)
    PyTorch:       0.2018 ms  (vs Triton:  3.52x)

Huge batch: 20000 segs, dim=128, avg_len=15, total=290421
  → Strategy: PyTorch

  Results:
    Triton:        0.0589 ms
    CUDA kernel:   0.0670 ms  (vs Triton:  0.88x)
    PyTorch:       0.2447 ms  (vs Triton:  4.16x)

================================================================================
Backward Performance Benchmarking
================================================================================

Small segs: 1000 segs, dim=128, avg_len=10, total=10096
  Results:
    Triton:    0.0805 ms
    PyTorch:   0.1958 ms  (ratio:  2.43x)
    Diff: 1.19e-07 ✓

Medium segs: 1000 segs, dim=256, avg_len=50, total=49767
  Results:
    Triton:    0.0908 ms
    PyTorch:   0.2023 ms  (ratio:  2.23x)
    Diff: 7.45e-09 ✓

Large segs: 500 segs, dim=256, avg_len=150, total=74505
  Results:
    Triton:    0.0956 ms
    PyTorch:   0.2496 ms  (ratio:  2.61x)
    Diff: 1.86e-09 ✓

Very large: 100 segs, dim=512, avg_len=600, total=59989
  Results:
    Triton:    0.1120 ms
    PyTorch:   0.3476 ms  (ratio:  3.10x)
    Diff: 4.66e-10 ✓

================================================================================
Complete Forward + Backward Benchmarking
================================================================================

Medium: 1000 segs, dim=256, total=49514
  Forward:   0.2320 ms
  Backward:  1.4158 ms
  Total:     1.6478 ms
  Backward/Forward ratio: 6.10x

Large: 500 segs, dim=512, total=49500
  Forward:   0.2111 ms
  Backward:  1.4425 ms
  Total:     1.6536 ms
  Backward/Forward ratio: 6.83x

Many segments: 10000 segs, dim=128, total=196033
  Forward:   0.2134 ms
  Backward:  1.5056 ms
  Total:     1.7189 ms
  Backward/Forward ratio: 7.06x

Mixed lengths: 1000 segs, dim=128, total=50139
  Forward:   0.2300 ms
  Backward:  1.4230 ms
  Total:     1.6530 ms
  Backward/Forward ratio: 6.19x

```
### Benchmark(1D grid)
```
Small segs: 1000 segs, dim=128, avg_len=10, total=10096
  Results:
    Triton:    0.8943 ms
    PyTorch:   1.6809 ms  (ratio:  1.88x)
    Diff: 1.19e-07 ✓

Medium segs: 1000 segs, dim=256, avg_len=50, total=49767
  Results:
    Triton:    1.0001 ms
    PyTorch:   1.6645 ms  (ratio:  1.66x)
    Diff: 7.45e-09 ✓

Large segs: 500 segs, dim=256, avg_len=150, total=74505
  Results:
    Triton:    0.8936 ms
    PyTorch:   1.6230 ms  (ratio:  1.82x)
    Diff: 1.86e-09 ✓

Very large: 100 segs, dim=512, avg_len=600, total=59989
  Results:
    Triton:    0.8731 ms
    PyTorch:   1.6304 ms  (ratio:  1.87x)
    Diff: 4.66e-10 ✓

Many small: 10000 segs, dim=128, avg_len=20, total=195425
  Results:
    Triton:    0.8999 ms
    PyTorch:   1.6412 ms  (ratio:  1.82x)
    Diff: 2.98e-08 ✓


================================================================================
Complete Forward + Backward Benchmarking
================================================================================

Medium: 1000 segs, dim=256, total=49514
  Forward:   0.8331 ms
  Backward:  3.7943 ms
  Total:     4.6274 ms
  Backward/Forward ratio: 4.55x

Large: 500 segs, dim=512, total=49500
  Forward:   0.8525 ms
  Backward:  3.8495 ms
  Total:     4.7020 ms
  Backward/Forward ratio: 4.52x
```

## Forward result(new)
### Benchmark with autotune
```
Small segs: 1000 segs, dim=128, avg_len=10, total=10074
  → Strategy: Triton

  Results:
    Triton:        0.0398 ms
    CUDA kernel:   0.0057 ms  (vs Triton:  6.98x)
    PyTorch:       0.1487 ms  (vs Triton:  3.73x)

Medium segs: 1000 segs, dim=256, avg_len=50, total=49247
  → Strategy: Triton

  Results:
    Triton:        0.0395 ms
    CUDA kernel:   0.0371 ms  (vs Triton:  1.06x)
    PyTorch:       0.1538 ms  (vs Triton:  3.90x)

Large segs: 500 segs, dim=256, avg_len=150, total=74682
  → Strategy: Triton

  Results:
    Triton:        0.0498 ms
    CUDA kernel:   0.0529 ms  (vs Triton:  0.94x)
    PyTorch:       0.1836 ms  (vs Triton:  3.69x)

Very large: 100 segs, dim=512, avg_len=600, total=60021
  → Strategy: Triton

  Results:
    Triton:        0.0736 ms
    CUDA kernel:   0.1256 ms  (vs Triton:  0.59x)
    PyTorch:       0.3117 ms  (vs Triton:  4.24x)

Many small: 10000 segs, dim=128, avg_len=20, total=196420
  → Strategy: PyTorch

  Results:
    Triton:        0.0659 ms
    CUDA kernel:   0.0678 ms  (vs Triton:  0.97x)
    PyTorch:       0.2135 ms  (vs Triton:  3.24x)

Huge batch: 20000 segs, dim=128, avg_len=15, total=289247
  → Strategy: PyTorch

  Results:
    Triton:        0.0950 ms
    CUDA kernel:   0.0989 ms  (vs Triton:  0.96x)
    PyTorch:       0.2777 ms  (vs Triton:  2.92x)
```

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