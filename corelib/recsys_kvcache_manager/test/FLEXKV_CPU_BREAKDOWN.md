# Recsys FlexKVCache Manager — CPU Breakdown Analysis

## Test Environment

GPU: **NVIDIA H100** 


## Micro-bench 1: 3-Step Pipeline + Two-Level Breakdown

### 1.1 Configuration

| Parameter | Value |
| --------- | ----- |
| `num_layers` | **8** |
| `head_dim` | **256** |
| `num_heads` | **4** |
| `batch_size` | **8** |
| `len_per_seq` / `sequence_length` | **1024 / 2048 / 4096** |
| `page_size` | **32** |
| `dtype` | **bf16** |

```text
[Step1: offload round (new sequence, no cache)] input (seq_len x batch_size) -> lookup -> allocate(+gpu.put) -> offload_launch(+put_async) -> offload_wait
[Step2] evict_gpu
[Step3: onboard round (100% GPU miss, 100% CPU hit)] input (the same, seq_len x batch_size) -> lookup -> allocate -> onboard_launch -> onboard_wait
```

### 1.2 L1/L2 latency breakdown

<table>
  <thead>
    <tr>
      <th rowspan="2">step</th>
      <th colspan="4">L1 (step-op, ms)</th>
      <th colspan="4">L2 (function, ms)</th>
    </tr>
    <tr>
      <th>op</th>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
      <th>function</th>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="12">offload round (new sequence, no cache)</td>
      <td rowspan="5">lookup</td>
      <td rowspan="5">2.834</td>
      <td rowspan="5">3.523</td>
      <td rowspan="5">3.642</td>
      <td><code>recsys.gpu.lookup</code></td>
      <td>0.088</td>
      <td>0.135</td>
      <td>0.143</td>
    </tr>
    <tr><td><code>recsys.host.build_index_meta</code></td><td>0.302</td><td>0.428</td><td>0.493</td></tr>
    <tr><td><code>recsys.host.adapter.to_get_match_requests</code></td><td>0.055</td><td>0.090</td><td>0.078</td></tr>
    <tr><td><code>flexkv.client.get_match</code></td><td>1.779</td><td>2.214</td><td>2.257</td></tr>
    <tr><td><code>recsys.merge_lookup_results</code></td><td>0.266</td><td>0.247</td><td>0.258</td></tr>
    <tr><td>allocate</td><td>0.839</td><td>1.140</td><td>1.153</td><td><code>recsys.gpu.allocate</code></td><td>0.825</td><td>1.126</td><td>1.140</td></tr>
    <tr><td rowspan="3">offload_launch</td><td rowspan="3">3.132</td><td rowspan="3">3.793</td><td rowspan="3">4.314</td><td><code>recsys.gpu.acquire_offload_pages</code></td><td>0.098</td><td>0.093</td><td>0.102</td></tr>
    <tr><td><code>recsys.host._build_slot_mappings</code></td><td>0.404</td><td>0.489</td><td>0.981</td></tr>
    <tr><td><code>flexkv.client.put_async</code></td><td>2.261</td><td>2.743</td><td>2.805</td></tr>
    <tr><td rowspan="3"><strong>offload_wait</strong></td><td rowspan="3"><strong>25.514</strong></td><td rowspan="3"><strong>30.655</strong></td><td rowspan="3"><strong>39.922</strong></td><td><code>recsys.host.offload_wait</code> (multiple <code>client.try_wait</code>)</td><td>13.082</td><td>15.180</td><td>18.577</td></tr>
    <tr><td><code>recsys.host.finish_task</code></td><td>0.010</td><td>0.010</td><td>0.011</td></tr>
    <tr><td><code>recsys.gpu.release_offload_pages</code></td><td>0.058</td><td>0.074</td><td>0.061</td></tr>
    <tr>
      <td rowspan="9">onboard round (100% GPU miss, 100% CPU hit)</td>
      <td rowspan="5">lookup</td>
      <td rowspan="5">1.830</td>
      <td rowspan="5">1.951</td>
      <td rowspan="5">1.997</td>
      <td><code>recsys.gpu.lookup</code></td><td>0.016</td><td>0.014</td><td>0.014</td>
    </tr>
    <tr><td><code>recsys.host.build_index_meta</code></td><td>0.090</td><td>0.106</td><td>0.105</td></tr>
    <tr><td><code>recsys.host.adapter.to_get_match_requests</code></td><td>0.031</td><td>0.031</td><td>0.029</td></tr>
    <tr><td><code>flexkv.client.get_match</code></td><td>1.260</td><td>1.380</td><td>1.418</td></tr>
    <tr><td><code>recsys.merge_lookup_results</code></td><td>0.201</td><td>0.200</td><td>0.202</td></tr>
    <tr><td>allocate</td><td>0.275</td><td>0.292</td><td>0.288</td><td><code>recsys.gpu.allocate</code></td><td>0.267</td><td>0.285</td><td>0.281</td></tr>
    <tr><td rowspan="2">onboard_launch</td><td rowspan="2">1.306</td><td rowspan="2">1.380</td><td rowspan="2">1.424</td><td><code>recsys.host._build_slot_mappings</code></td><td>0.630</td><td>0.623</td><td>0.615</td></tr>
    <tr><td><code>flexkv.client.launch</code></td><td>0.259</td><td>0.311</td><td>0.324</td></tr>
    <tr><td><strong>onboard_wait</strong></td><td><strong>17.341</strong></td><td><strong>21.244</strong></td><td><strong>37.272</strong></td><td><code>flexkv.client.wait</code></td><td>17.270</td><td>21.180</td><td>37.143</td></tr>
  </tbody>
</table>

### 1.3 L1/L2 latency breakdown percentage (`seq_len=2048`)

Inner ring: **L1** step-ops (lookup, allocate, onboard launch, onboard wait, offload launch, offload wait).
Outer ring: **L2** functions nested within the corresponding L1 sector.
Outer-ring **other overhead** is the remaining L1 time not attributed to the listed nested timers.
The denominator excludes `evict_gpu` and the Step3 lookup/allocate stages.

<p align="center">
  <img src="breakdown_1/H100_result/L1L2_latency_breakdown_latency_bs8_len2048.png" alt="L1/L2 latency breakdown percentage" width="620"/>
</p>

### 1.4 Conclusion

- **Transfer-related latency remains the dominant component** of the measured KV-cache pipeline latency.
- The remaining latency comes from metadata preparation, lookup, allocation, offload/onboard launch, and wrapper/runtime bookkeeping.
- As `seq_len` increases, fixed metadata/orchestration costs become relatively smaller, so the pipeline scales primarily with transfer volume and bandwidth overlap.

---

## Micro-bench 2: `as_batch` Performance

The following results use the same configuration as Micro-bench 1 and compare `flexkv_as_batch=0` with `flexkv_as_batch=1`.

### 2.1 Launch + Wait Summary

<table>
  <thead>
    <tr>
      <th>path</th>
      <th>seq_len</th>
      <th><code>as_batch=0</code> (ms)</th>
      <th><code>as_batch=1</code> (ms)</th>
      <th>delta</th>
    </tr>
  </thead>
  <tbody>
    <tr><td rowspan="3">offload</td><td>1024</td><td>26.87</td><td>18.63</td><td><strong>-30.7%</strong></td></tr>
    <tr><td>2048</td><td>30.59</td><td>25.77</td><td><strong>-15.7%</strong></td></tr>
    <tr><td>4096</td><td>39.17</td><td>38.58</td><td>-1.5%</td></tr>
    <tr><td rowspan="3">onboard</td><td>1024</td><td>17.51</td><td>14.65</td><td><strong>-16.4%</strong></td></tr>
    <tr><td>2048</td><td>26.31</td><td>20.04</td><td><strong>-23.9%</strong></td></tr>
    <tr><td>4096</td><td>40.10</td><td>30.59</td><td><strong>-23.7%</strong></td></tr>
  </tbody>
</table>

---

## Micro-bench 3: Offload Stress — Effective KV Bandwidth

This stress test compares `as_batch=0` and `as_batch=1` in a burst of `offload_launch x N` followed by one drain of `offload_try_wait`.

### 3.1 Configuration

| Parameter | Value |
| --- | --- |
| `num_layers` | **64** |
| `num_kv_heads` | **8** |
| `head_dim` | **256** |
| `dtype` | **bf16** |
| Peak reference BW (H100 D2H) | **64 GiB/s** |

Effective KV bandwidth is computed as real KV payload divided by total burst drain time.

### 3.2 Results

| scenario | launch_count | `as_batch=0` eff. GiB/s | `as_batch=1` eff. GiB/s | delta | util vs 64 GiB/s |
| --- | --- | --- | --- | --- | --- |
| bs=1, seq=1024 | 50 | 44.4 | 44.3 | -0.3% | 69.4% |
| bs=1, seq=1024 | 100 | 44.4 | 45.0 | +1.4% | 70.3% |
| bs=1, seq=1024 | 150 | 46.1 | 45.5 | -1.2% | 71.1% |
| bs=4, seq=512 | 5 | 33.7 | 36.8 | **+9.2%** | 57.5% |
| bs=4, seq=512 | 10 | 41.6 | 44.9 | **+7.9%** | 70.2% |
| bs=4, seq=512 | 15 | 43.1 | 45.1 | **+4.6%** | 70.5% |
| bs=4, seq=512 | 20 | 42.5 | 45.3 | **+6.6%** | 70.8% |

### 3.3 Conclusion

- bs=1 does not trigger the multi-user batch path and shows no regression.
- bs=4 triggers `as_batch` and improves effective KV bandwidth by **5-9%**, reaching about **71%** of the 64 GiB/s reference bandwidth.
