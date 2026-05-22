# Recsys KVCache Manager — Performance Analysis

---

## Test Environment


| Item       | Value                                               |
| ---------- | --------------------------------------------------- |
| GPU        | **NVIDIA L20**                                      |
| Node       | **a1u1g-rome-0055**                                 |
| KV tensors | `bf16`, shape `(num_layers=3, max_seq_len, 4, 128)` |


---

## Micro-bench 1: 3-Step Pipeline + Three-Level Breakdown

### 1.1 Configuration


| Parameter                         | Value                  |
| --------------------------------- | ---------------------- |
| `batch_size`                      | **8**                  |
| `len_per_seq` / `sequence_length` | **1024 / 2048 / 4096** |


```text
[Step1: offload round（100% GPU miss）] input (1024 x 8) → lookup → allocate(+gpu.put) → offload_launch(+put_async) → offload_wait
[Step2] evict_gpu
[Step3: onboard round (100% GPU hit)] input (the same, 1024 x 8 ) → lookup → allocate → onboard_launch → onboard_wait
```

### 1.2 step-op results


<table>
  <thead>
    <tr>
      <th>step</th>
      <th>op</th>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">offload round（100% GPU miss）</td>
      <td>lookup</td>
      <td>10.18</td>
      <td>5.04</td>
      <td>5.09</td>
    </tr>
    <tr>
      <td>allocate</td>
      <td>9.81</td>
      <td>1.15</td>
      <td>1.14</td>
    </tr>
    <tr>
      <td>offload_launch</td>
      <td>11.44</td>
      <td>9.74</td>
      <td>10.25</td>
    </tr>
    <tr>
      <td><strong>offload_wait</strong></td>
      <td><strong>15.34</strong></td>
      <td><strong>24.48</strong></td>
      <td><strong>39.41</strong></td>
    </tr>
    <tr>
      <td>evict</td>
      <td>evict_gpu</td>
      <td>0.11</td>
      <td>0.10</td>
      <td>0.10</td>
    </tr>
    <tr>
      <td rowspan="4">onboard round（100% GPU hit）</td>
      <td>lookup</td>
      <td>3.44</td>
      <td>3.14</td>
      <td>3.50</td>
    </tr>
    <tr>
      <td>allocate</td>
      <td>0.46</td>
      <td>0.44</td>
      <td>0.46</td>
    </tr>
    <tr>
      <td>onboard_launch</td>
      <td>2.77</td>
      <td>2.29</td>
      <td>2.30</td>
    </tr>
    <tr>
      <td>onboard_wait</td>
      <td>9.27</td>
      <td>10.69</td>
      <td>15.78</td>
    </tr>
  </tbody>
</table>


![Step ops breakdown](breakdown_1/profiler_result/plot/plot/step_op_flow_bs8.png)

### 1.3 L3 key-call breakdown


<table>
  <thead>
    <tr>
      <th>step</th>
      <th>op</th>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="13">offload round（100% GPU miss）</td>
      <td><code>gpu.lookup_py</code></td>
      <td>0.114</td>
      <td>0.109</td>
      <td>0.120</td>
    </tr>
    <tr>
      <td><code>flexkv.build_index_meta</code></td>
      <td>1.849</td>
      <td>0.458</td>
      <td>0.487</td>
    </tr>
    <tr>
      <td><code>flexkv.adapter.to_get_match_requests</code></td>
      <td>0.139</td>
      <td>0.136</td>
      <td>0.144</td>
    </tr>
    <tr>
      <td><code>flexkv.client.get_match</code></td>
      <td>6.791</td>
      <td>4.674</td>
      <td>5.059</td>
    </tr>
    <tr>
      <td><code>recsys.merge_lookup_results</code></td>
      <td>2.749</td>
      <td>0.788</td>
      <td>0.804</td>
    </tr>
    <tr>
      <td><code>gpu.allocate_py</code></td>
      <td>16.984</td>
      <td>1.909</td>
      <td>1.937</td>
    </tr>
    <tr>
      <td><code>gpu.acquire_offload_pages_py</code></td>
      <td>0.230</td>
      <td>0.213</td>
      <td>0.219</td>
    </tr>
    <tr>
      <td><code>flexkv._build_slot_mappings</code></td>
      <td>8.096</td>
      <td>5.630</td>
      <td>5.645</td>
    </tr>
    <tr>
      <td><code>flexkv.client.put_async</code></td>
      <td>3.547</td>
      <td>3.994</td>
      <td>4.346</td>
    </tr>
    <tr>
      <td><code>flexkv.client.launch</code></td>
      <td>0.404</td>
      <td>0.401</td>
      <td>0.382</td>
    </tr>
    <tr>
      <td><code>flexkv.client.try_wait</code></td>
      <td>7.681</td>
      <td>12.137</td>
      <td>18.742</td>
    </tr>
    <tr>
      <td><code>flexkv.finish_task</code></td>
      <td>0.417</td>
      <td>0.439</td>
      <td>0.417</td>
    </tr>
    <tr>
      <td><code>gpu.release_offload_pages_py</code></td>
      <td>0.224</td>
      <td>0.228</td>
      <td>0.163</td>
    </tr>
    <tr>
      <td>evict</td>
      <td><code>gpu.evict_py</code></td>
      <td>0.126</td>
      <td>0.119</td>
      <td>0.113</td>
    </tr>
    <tr>
      <td rowspan="10">onboard round（100% GPU hit）</td>
      <td><code>gpu.lookup_py</code></td>
      <td>0.103</td>
      <td>0.101</td>
      <td>0.105</td>
    </tr>
    <tr>
      <td><code>flexkv.build_index_meta</code></td>
      <td>1.849</td>
      <td>0.458</td>
      <td>0.487</td>
    </tr>
    <tr>
      <td><code>flexkv.adapter.to_get_match_requests</code></td>
      <td>0.139</td>
      <td>0.136</td>
      <td>0.144</td>
    </tr>
    <tr>
      <td><code>flexkv.client.get_match</code></td>
      <td>6.791</td>
      <td>4.674</td>
      <td>5.059</td>
    </tr>
    <tr>
      <td><code>recsys.merge_lookup_results</code></td>
      <td>2.749</td>
      <td>0.788</td>
      <td>0.804</td>
    </tr>
    <tr>
      <td><code>gpu.allocate_py</code></td>
      <td>10.404</td>
      <td>1.713</td>
      <td>1.738</td>
    </tr>
    <tr>
      <td><code>flexkv._build_slot_mappings</code></td>
      <td>8.096</td>
      <td>5.630</td>
      <td>5.645</td>
    </tr>
    <tr>
      <td><code>flexkv.client.put_async</code></td>
      <td>3.547</td>
      <td>3.994</td>
      <td>4.346</td>
    </tr>
    <tr>
      <td><code>flexkv.client.launch</code></td>
      <td>0.404</td>
      <td>0.401</td>
      <td>0.382</td>
    </tr>
    <tr>
      <td><code>flexkv.client.wait</code></td>
      <td>9.567</td>
      <td>11.002</td>
      <td>16.082</td>
    </tr>
  </tbody>
</table>


### 1.4 Offload_try_wait single breakdown

Purpose: we observed a large gap between accumulated `try_wait` time and `wait` time, so this section isolates `offload_try_wait` behavior for a dedicated decomposition.

#### A. Try_wait Breakdown

The results below show averages over 9 trials, excluding the first trial.


| `len_per_seq` | mode     | avg `T_try_wait` total (ms) | avg `try_wait` calls | avg `~T_try_wait/call` (ms) | avg `T_wait` total (ms) | avg `wait` calls | avg `~T_wait/call` (ms) | avg decomposition `T_try_wait x N + T_wait` (ms) | avg offload loop wall (ms) |
| ------------- | -------- | --------------------------- | -------------------- | --------------------------- | ----------------------- | ---------------- | ----------------------- | ------------------------------------------------ | -------------------------- |
| 1024          | baseline | 7.548                       | 320.4                | ~0.024                      | 0.357                   | 1.0              | ~0.357                  | 7.905                                            | 10.826                     |
| 2048          | baseline | 13.621                      | 586.6                | ~0.023                      | 0.352                   | 1.0              | ~0.352                  | 13.973                                           | 19.223                     |
| 4096          | baseline | 26.696                      | 1100.2               | ~0.024                      | 0.337                   | 1.0              | ~0.337                  | 27.033                                           | 36.439                     |


#### B. `baseline` vs `wait_only`

- `baseline`: using try_wait + wait, i.e., `N` rounds of `client.try_wait` + one final `client.wait`.
- `wait_only`: using wait only, i.e., `client.wait`.


| mode      | avg `try_wait` calls | avg `wait` calls | avg `T_try_wait` total (ms) | avg `~T_try_wait/call` (ms) | avg `T_wait` total (ms) | avg `~T_wait/call` (ms) | avg decomposition (ms) | avg offload loop wall (ms) |
| --------- | -------------------- | ---------------- | --------------------------- | --------------------------- | ----------------------- | ----------------------- | ---------------------- | -------------------------- |
| baseline  | 320.4                | 1.0              | 7.548                       | ~0.024                      | 0.357                   | ~0.357                  | 7.905                  | 10.826                     |
| wait_only | 0.0                  | 1.0              | 0.000                       | ~0.000                      | 10.301                  | ~10.301                 | 10.301                 | 10.441                     |


#### C. Conclusion

- Single-call `try_wait` latency remains relatively stable at about `~0.023-0.025 ms`.
- For the updated `1024` / `2048` / `4096` runs, `try_wait` accounts for `70.2%` / `71.4%` / `73.7%` of total `offload_try_wait` time, respectively.

---

## Micro-bench 2: Incremental Offload Stress (`launch x N` + one `offload_try_wait`)

### 2.1 Experimental setup

- stress loop pattern:
  1. run `launch_count = N` rounds of `put_async` (each round submits one full batch);
  2. then run **one** explicit `offload_try_wait` stage (internally polling `try_wait` until done or timeout).
- per-launch payload:
  - `batch_size = 8`
  - `sequence_length = 1024`
  - `launch_count = 50, 100, 150, ..., 400`.
  - `cpu_cache_gb = 80GB`

### 2.2 Results


| `launch_count` | expected tokens | success tokens | failed tasks | timeout tasks | `try_wait` rounds | `try_wait` time (ms) | data size (GB) | bandwidth (GB/s) |
| -------------- | --------------- | -------------- | ------------ | ------------- | ----------------- | -------------------- | -------------- | ---------------- |
| 50             | 409600          | 409600         | 0            | 0             | 67                | 711.30               | 4.688          | 6.59             |
| 100            | 819200          | 819200         | 0            | 0             | 124               | 1411.34              | 9.375          | 6.64             |
| 150            | 1228800         | 1228800        | 0            | 0             | 185               | 2254.60              | 14.062         | 6.24             |
| 200            | 1638400         | 1638400        | 0            | 0             | 221               | 2935.82              | 18.750         | 6.39             |
| 250            | 2048000         | 2048000        | 0            | 0             | 263               | 3779.90              | 23.438         | 6.20             |
| 300            | 2457600         | 2457600        | 0            | 0             | 306               | 4517.49              | 28.125         | 6.23             |
| 350            | 2867200         | 2867200        | 0            | 0             | 344               | 5297.92              | 32.812         | 6.19             |
| 400            | 3276800         | 3276800        | 0            | 0             | 374               | 6063.08              | 37.500         | 6.18             |


### 2.3 Conclusions

- **Reliability under stress:** all runs have `success_tokens == expected_tokens`, `failed_tasks = 0`, `timeout_tasks = 0`.
- **Stable throughput:** effective bandwidth stays in a narrow range (`6.18-6.64 GB/s`, around `~6.3 GB/s`) from `4.688 GB` to `37.500 GB`.
- **Scaling trend:** as `launch_count` increases (`50 -> 400`), cumulative `try_wait` calls (`67 -> 374`) and `try_wait` time (`711.30 -> 6063.08 ms`) increase accordingly, showing predictable stress scaling.

