# Recsys FlexKVCache Manager — CPU Breakdown Analysis (slotmapping + as_batch)

## Test Environment

| Item | Value |
| --- | --- |
| GPU | **NVIDIA H100** |
| Host | bare metal `10.176.171.85` |

---

## Micro-bench 1: 3-Step Pipeline + Two-Level Breakdown

### 1.1 Conclusion 1
Baseline: 没有做 slotmapping + as_batch 优化之前的

- **offload（相对 Baseline）**：`offload_launch` **−47% ~ −60%**（2048：7.19 → 3.35 ms）；`offload_wait` **−14% ~ −37%**（2048：30.66 → 22.42 ms）； launch+wait seq_len = 2048 **37.85 → 25.77 ms（−31.9%）**。主要收益来自 `_build_slot_mappings`（3.30 → 0.43 ms）与 as_batch 下 `finish_task`（D2H）绝对值下降。
- **onboard（相对 Baseline）**：`onboard_launch` **−48% ~ −50%**（2048：1.38 → 0.70 ms）；`onboard_wait` **−9% ~ −20%**（2048：21.24 → 19.34 ms）；launch+wait seq_len = 2048 **22.62 → 20.04 ms（−11.4%）**。batch `launch` 路径收益显著。
- **Note**：
  a. 短 seq 上 offload_wait 降幅更大，数据量合并带宽更高。
  b. 相对 slotmapping only，as_batch 在 offload_launch 上会用 `put_match×N + launch（batch_put）`（串行）替代 `put_async`，可能抬高 launch；但相对 **Before**，slot_mapping + as_batch 叠加后 launch 仍净降 **~50%**。
  c. 除 onboard + offload 外的 APIs 波动主要来自 **不同 profiling run 波动**。
  d. **（offload_wait L2 Δ%）**：以 **L1** 为准即可。

### 1.2 L1/L2 breakdown — Optimized（v2 + as_batch）

<table>
  <thead>
    <tr>
      <th rowspan="2">step</th>
      <th rowspan="2">L1 op</th>
      <th colspan="3">L1 (step-op, ms)</th>
      <th rowspan="2">L2 function</th>
      <th colspan="3">L2 (function, ms)</th>
    </tr>
    <tr>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="12">offload round (new sequence, no cache)</td>
      <td rowspan="5">lookup</td>
      <td rowspan="5">2.316</td>
      <td rowspan="5">3.267</td>
      <td rowspan="5">3.589</td>
      <td><code>recsys.gpu.lookup</code></td>
      <td>0.056</td>
      <td>0.134</td>
      <td>0.157</td>
    </tr>
    <tr>
      <td><code>recsys.host.build_index_meta</code></td>
      <td>0.256</td>
      <td>0.409</td>
      <td>0.461</td>
    </tr>
    <tr>
      <td><code>recsys.host.adapter.to_get_match_requests</code></td>
      <td>0.048</td>
      <td>0.080</td>
      <td>0.077</td>
    </tr>
    <tr>
      <td><code>flexkv.client.get_match</code></td>
      <td>1.462</td>
      <td>2.036</td>
      <td>2.242</td>
    </tr>
    <tr>
      <td><code>recsys.merge_lookup_results</code></td>
      <td>0.220</td>
      <td>0.257</td>
      <td>0.266</td>
    </tr>
    <tr>
      <td>allocate</td>
      <td>0.762</td>
      <td>1.128</td>
      <td>1.140</td>
      <td><code>recsys.gpu.allocate</code></td>
      <td>0.751</td>
      <td>1.114</td>
      <td>1.126</td>
    </tr>
    <tr>
      <td rowspan="3">offload_launch</td>
      <td rowspan="3">2.647</td>
      <td rowspan="3">3.345</td>
      <td rowspan="3">4.203</td>
      <td><code>recsys.gpu.acquire_offload_pages</code></td>
      <td>0.091</td>
      <td>0.101</td>
      <td>0.112</td>
    </tr>
    <tr>
      <td><code>recsys.host._build_slot_mappings</code></td>
      <td>0.375</td>
      <td>0.425</td>
      <td>0.836</td>
    </tr>
    <tr>
      <td><code>put_match + batch launch</code> (unlabeled)</td>
      <td>2.182</td>
      <td>2.819</td>
      <td>3.255</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>offload_wait</strong></td>
      <td rowspan="3"><strong>15.981</strong></td>
      <td rowspan="3"><strong>22.422</strong></td>
      <td rowspan="3"><strong>34.374</strong></td>
      <td><code>recsys.host.offload_wait</code></td>
      <td>0.109</td>
      <td>0.161</td>
      <td>0.159</td>
    </tr>
    <tr>
      <td><code>recsys.host.finish_task</code></td>
      <td>15.688</td>
      <td>22.006</td>
      <td>33.939</td>
    </tr>
    <tr>
      <td><code>recsys.gpu.release_offload_pages</code></td>
      <td>0.089</td>
      <td>0.126</td>
      <td>0.134</td>
    </tr>
    <tr>
      <td rowspan="9">onboard round (100% GPU miss, 100% CPU hit)</td>
      <td rowspan="5">lookup</td>
      <td rowspan="5">2.173</td>
      <td rowspan="5">3.607</td>
      <td rowspan="5">4.026</td>
      <td><code>recsys.gpu.lookup</code></td>
      <td>0.022</td>
      <td>0.032</td>
      <td>0.034</td>
    </tr>
    <tr>
      <td><code>recsys.host.build_index_meta</code></td>
      <td>0.116</td>
      <td>0.199</td>
      <td>0.206</td>
    </tr>
    <tr>
      <td><code>recsys.host.adapter.to_get_match_requests</code></td>
      <td>0.050</td>
      <td>0.074</td>
      <td>0.076</td>
    </tr>
    <tr>
      <td><code>flexkv.client.get_match</code></td>
      <td>1.522</td>
      <td>2.689</td>
      <td>3.070</td>
    </tr>
    <tr>
      <td><code>recsys.merge_lookup_results</code></td>
      <td>0.202</td>
      <td>0.203</td>
      <td>0.206</td>
    </tr>
    <tr>
      <td>allocate</td>
      <td>0.265</td>
      <td>0.279</td>
      <td>0.271</td>
      <td><code>recsys.gpu.allocate</code></td>
      <td>0.258</td>
      <td>0.272</td>
      <td>0.263</td>
    </tr>
    <tr>
      <td rowspan="2">onboard_launch</td>
      <td rowspan="2">0.660</td>
      <td rowspan="2">0.696</td>
      <td rowspan="2">0.747</td>
      <td><code>recsys.host._build_slot_mappings</code></td>
      <td>0.168</td>
      <td>0.176</td>
      <td>0.187</td>
    </tr>
    <tr>
      <td><code>flexkv.client.launch</code></td>
      <td>0.254</td>
      <td>0.270</td>
      <td>0.304</td>
    </tr>
    <tr>
      <td><strong>onboard_wait</strong></td>
      <td><strong>13.989</strong></td>
      <td><strong>19.341</strong></td>
      <td><strong>29.842</strong></td>
      <td><code>flexkv.client.wait</code></td>
      <td>13.873</td>
      <td>19.263</td>
      <td>29.765</td>
    </tr>
  </tbody>
</table>

> offload batch 路径：`flexkv.client.put_async` L2 消失；`put_match + batch launch` = L1 offload_launch 减去已标注 L2（主要为 `put_match×N` + batch `launch`，无单独 NVTX）。

#### 占 pipeline total 比例 (%)

<table>
  <thead>
    <tr>
      <th rowspan="2">step</th>
      <th rowspan="2">L1 op</th>
      <th colspan="3">L1 (step-op, %)</th>
      <th rowspan="2">L2 function</th>
      <th colspan="3">L2 (function, %)</th>
    </tr>
    <tr>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="12">offload round (new sequence, no cache)</td>
      <td rowspan="5">lookup</td>
      <td rowspan="5">5.97%</td>
      <td rowspan="5">6.04%</td>
      <td rowspan="5">4.59%</td>
      <td><code>recsys.gpu.lookup</code></td>
      <td>0.14%</td>
      <td>0.25%</td>
      <td>0.20%</td>
    </tr>
    <tr>
      <td><code>recsys.host.build_index_meta</code></td>
      <td>0.66%</td>
      <td>0.76%</td>
      <td>0.59%</td>
    </tr>
    <tr>
      <td><code>recsys.host.adapter.to_get_match_requests</code></td>
      <td>0.12%</td>
      <td>0.15%</td>
      <td>0.10%</td>
    </tr>
    <tr>
      <td><code>flexkv.client.get_match</code></td>
      <td>3.77%</td>
      <td>3.76%</td>
      <td>2.87%</td>
    </tr>
    <tr>
      <td><code>recsys.merge_lookup_results</code></td>
      <td>0.57%</td>
      <td>0.48%</td>
      <td>0.34%</td>
    </tr>
    <tr>
      <td>allocate</td>
      <td>1.96%</td>
      <td>2.09%</td>
      <td>1.46%</td>
      <td><code>recsys.gpu.allocate</code></td>
      <td>1.94%</td>
      <td>2.06%</td>
      <td>1.44%</td>
    </tr>
    <tr>
      <td rowspan="3">offload_launch</td>
      <td rowspan="3">6.82%</td>
      <td rowspan="3">6.19%</td>
      <td rowspan="3">5.38%</td>
      <td><code>recsys.gpu.acquire_offload_pages</code></td>
      <td>0.23%</td>
      <td>0.19%</td>
      <td>0.14%</td>
    </tr>
    <tr>
      <td><code>recsys.host._build_slot_mappings</code></td>
      <td>0.97%</td>
      <td>0.79%</td>
      <td>1.07%</td>
    </tr>
    <tr>
      <td><code>put_match + batch launch</code> (unlabeled)</td>
      <td>5.62%</td>
      <td>5.21%</td>
      <td>4.16%</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>offload_wait</strong></td>
      <td rowspan="3"><strong>41.20%</strong></td>
      <td rowspan="3"><strong>41.46%</strong></td>
      <td rowspan="3"><strong>43.96%</strong></td>
      <td><code>recsys.host.offload_wait</code></td>
      <td>0.28%</td>
      <td>0.30%</td>
      <td>0.20%</td>
    </tr>
    <tr>
      <td><code>recsys.host.finish_task</code></td>
      <td>40.44%</td>
      <td>40.69%</td>
      <td>43.40%</td>
    </tr>
    <tr>
      <td><code>recsys.gpu.release_offload_pages</code></td>
      <td>0.23%</td>
      <td>0.23%</td>
      <td>0.17%</td>
    </tr>
    <tr>
      <td rowspan="9">onboard round (100% GPU miss, 100% CPU hit)</td>
      <td rowspan="5">lookup</td>
      <td rowspan="5">5.60%</td>
      <td rowspan="5">6.67%</td>
      <td rowspan="5">5.15%</td>
      <td><code>recsys.gpu.lookup</code></td>
      <td>0.06%</td>
      <td>0.06%</td>
      <td>0.04%</td>
    </tr>
    <tr>
      <td><code>recsys.host.build_index_meta</code></td>
      <td>0.30%</td>
      <td>0.37%</td>
      <td>0.26%</td>
    </tr>
    <tr>
      <td><code>recsys.host.adapter.to_get_match_requests</code></td>
      <td>0.13%</td>
      <td>0.14%</td>
      <td>0.10%</td>
    </tr>
    <tr>
      <td><code>flexkv.client.get_match</code></td>
      <td>3.92%</td>
      <td>4.97%</td>
      <td>3.93%</td>
    </tr>
    <tr>
      <td><code>recsys.merge_lookup_results</code></td>
      <td>0.52%</td>
      <td>0.38%</td>
      <td>0.26%</td>
    </tr>
    <tr>
      <td>allocate</td>
      <td>0.68%</td>
      <td>0.52%</td>
      <td>0.35%</td>
      <td><code>recsys.gpu.allocate</code></td>
      <td>0.66%</td>
      <td>0.50%</td>
      <td>0.34%</td>
    </tr>
    <tr>
      <td rowspan="2">onboard_launch</td>
      <td rowspan="2">1.70%</td>
      <td rowspan="2">1.29%</td>
      <td rowspan="2">0.96%</td>
      <td><code>recsys.host._build_slot_mappings</code></td>
      <td>0.43%</td>
      <td>0.32%</td>
      <td>0.24%</td>
    </tr>
    <tr>
      <td><code>flexkv.client.launch</code></td>
      <td>0.65%</td>
      <td>0.50%</td>
      <td>0.39%</td>
    </tr>
    <tr>
      <td><strong>onboard_wait</strong></td>
      <td><strong>36.06%</strong></td>
      <td><strong>35.76%</strong></td>
      <td><strong>38.17%</strong></td>
      <td><code>flexkv.client.wait</code></td>
      <td>35.76%</td>
      <td>35.62%</td>
      <td>38.07%</td>
    </tr>
  </tbody>
</table>

### 1.3 L1/L2 breakdown — 相对 baseline 的变化（Δ = Optimized − Baseline）

#### Δ 绝对值 (ms)

<table>
  <thead>
    <tr>
      <th rowspan="2">step</th>
      <th rowspan="2">L1 op</th>
      <th colspan="3">L1 Δ (ms)</th>
      <th rowspan="2">L2 function</th>
      <th colspan="3">L2 Δ (ms)</th>
    </tr>
    <tr>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="12">offload round (new sequence, no cache)</td>
      <td rowspan="5">lookup</td>
      <td rowspan="5">−0.518</td>
      <td rowspan="5">−0.256</td>
      <td rowspan="5">−0.053</td>
      <td><code>recsys.gpu.lookup</code></td>
      <td>−0.032</td>
      <td>−0.001</td>
      <td>+0.014</td>
    </tr>
    <tr>
      <td><code>recsys.host.build_index_meta</code></td>
      <td>−0.046</td>
      <td>−0.019</td>
      <td>−0.032</td>
    </tr>
    <tr>
      <td><code>recsys.host.adapter.to_get_match_requests</code></td>
      <td>−0.007</td>
      <td>−0.010</td>
      <td>−0.001</td>
    </tr>
    <tr>
      <td><code>flexkv.client.get_match</code></td>
      <td>−0.317</td>
      <td>−0.178</td>
      <td>−0.015</td>
    </tr>
    <tr>
      <td><code>recsys.merge_lookup_results</code></td>
      <td>−0.046</td>
      <td>+0.010</td>
      <td>+0.008</td>
    </tr>
    <tr>
      <td>allocate</td>
      <td>−0.077</td>
      <td>−0.012</td>
      <td>−0.013</td>
      <td><code>recsys.gpu.allocate</code></td>
      <td>−0.074</td>
      <td>−0.012</td>
      <td>−0.014</td>
    </tr>
    <tr>
      <td rowspan="3">offload_launch</td>
      <td rowspan="3">−4.035</td>
      <td rowspan="3">−3.847</td>
      <td rowspan="3">−3.745</td>
      <td><code>recsys.gpu.acquire_offload_pages</code></td>
      <td>−0.007</td>
      <td>+0.008</td>
      <td>+0.010</td>
    </tr>
    <tr>
      <td><code>recsys.host._build_slot_mappings</code></td>
      <td>−2.954</td>
      <td>−2.878</td>
      <td>−2.954</td>
    </tr>
    <tr>
      <td><code>put_match + batch launch</code> (vs <code>put_async</code>)</td>
      <td>−0.347</td>
      <td>−0.092</td>
      <td>+0.063</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>offload_wait</strong></td>
      <td rowspan="3"><strong>−9.533</strong></td>
      <td rowspan="3"><strong>−8.233</strong></td>
      <td rowspan="3"><strong>−5.548</strong></td>
      <td><code>recsys.host.offload_wait</code></td>
      <td>−12.973</td>
      <td>−15.019</td>
      <td>−18.418</td>
    </tr>
    <tr>
      <td><code>recsys.host.finish_task</code></td>
      <td>+15.678</td>
      <td>+21.996</td>
      <td>+33.928</td>
    </tr>
    <tr>
      <td><code>recsys.gpu.release_offload_pages</code></td>
      <td>+0.031</td>
      <td>+0.052</td>
      <td>+0.073</td>
    </tr>
    <tr>
      <td rowspan="9">onboard round (100% GPU miss, 100% CPU hit)</td>
      <td rowspan="5">lookup</td>
      <td rowspan="5">+0.343</td>
      <td rowspan="5">+1.656</td>
      <td rowspan="5">+2.029</td>
      <td><code>recsys.gpu.lookup</code></td>
      <td>+0.006</td>
      <td>+0.018</td>
      <td>+0.020</td>
    </tr>
    <tr>
      <td><code>recsys.host.build_index_meta</code></td>
      <td>+0.026</td>
      <td>+0.093</td>
      <td>+0.101</td>
    </tr>
    <tr>
      <td><code>recsys.host.adapter.to_get_match_requests</code></td>
      <td>+0.019</td>
      <td>+0.043</td>
      <td>+0.047</td>
    </tr>
    <tr>
      <td><code>flexkv.client.get_match</code></td>
      <td>+0.262</td>
      <td>+1.309</td>
      <td>+1.652</td>
    </tr>
    <tr>
      <td><code>recsys.merge_lookup_results</code></td>
      <td>+0.001</td>
      <td>+0.003</td>
      <td>+0.004</td>
    </tr>
    <tr>
      <td>allocate</td>
      <td>−0.010</td>
      <td>−0.013</td>
      <td>−0.017</td>
      <td><code>recsys.gpu.allocate</code></td>
      <td>−0.009</td>
      <td>−0.013</td>
      <td>−0.018</td>
    </tr>
    <tr>
      <td rowspan="2">onboard_launch</td>
      <td rowspan="2">−0.646</td>
      <td rowspan="2">−0.684</td>
      <td rowspan="2">−0.677</td>
      <td><code>recsys.host._build_slot_mappings</code></td>
      <td>−0.462</td>
      <td>−0.447</td>
      <td>−0.428</td>
    </tr>
    <tr>
      <td><code>flexkv.client.launch</code></td>
      <td>−0.005</td>
      <td>−0.041</td>
      <td>−0.020</td>
    </tr>
    <tr>
      <td><strong>onboard_wait</strong></td>
      <td><strong>−3.352</strong></td>
      <td><strong>−1.903</strong></td>
      <td><strong>−7.430</strong></td>
      <td><code>flexkv.client.wait</code></td>
      <td>−3.397</td>
      <td>−1.917</td>
      <td>−7.378</td>
    </tr>
  </tbody>
</table>

#### Δ 相对 baseline (%)

<table>
  <thead>
    <tr>
      <th rowspan="2">step</th>
      <th rowspan="2">L1 op</th>
      <th colspan="3">L1 Δ (%)</th>
      <th rowspan="2">L2 function</th>
      <th colspan="3">L2 Δ (%)</th>
    </tr>
    <tr>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="12">offload round (new sequence, no cache)</td>
      <td rowspan="5">lookup</td>
      <td rowspan="5">−18.3%</td>
      <td rowspan="5">−7.3%</td>
      <td rowspan="5">−1.5%</td>
      <td><code>recsys.gpu.lookup</code></td>
      <td>−36.4%</td>
      <td>−0.7%</td>
      <td>+9.8%</td>
    </tr>
    <tr>
      <td><code>recsys.host.build_index_meta</code></td>
      <td>−15.2%</td>
      <td>−4.4%</td>
      <td>−6.5%</td>
    </tr>
    <tr>
      <td><code>recsys.host.adapter.to_get_match_requests</code></td>
      <td>−12.7%</td>
      <td>−11.1%</td>
      <td>−1.3%</td>
    </tr>
    <tr>
      <td><code>flexkv.client.get_match</code></td>
      <td>−17.8%</td>
      <td>−8.0%</td>
      <td>−0.7%</td>
    </tr>
    <tr>
      <td><code>recsys.merge_lookup_results</code></td>
      <td>−17.3%</td>
      <td>+4.0%</td>
      <td>+3.1%</td>
    </tr>
    <tr>
      <td>allocate</td>
      <td>−9.2%</td>
      <td>−1.1%</td>
      <td>−1.1%</td>
      <td><code>recsys.gpu.allocate</code></td>
      <td>−9.0%</td>
      <td>−1.1%</td>
      <td>−1.2%</td>
    </tr>
    <tr>
      <td rowspan="3">offload_launch</td>
      <td rowspan="3">−60.4%</td>
      <td rowspan="3">−53.5%</td>
      <td rowspan="3">−47.1%</td>
      <td><code>recsys.gpu.acquire_offload_pages</code></td>
      <td>−7.1%</td>
      <td>+8.6%</td>
      <td>+9.8%</td>
    </tr>
    <tr>
      <td><code>recsys.host._build_slot_mappings</code></td>
      <td>−88.7%</td>
      <td>−87.1%</td>
      <td>−77.9%</td>
    </tr>
    <tr>
      <td><code>put_match + batch launch</code> (vs <code>put_async</code>)</td>
      <td>−13.7%</td>
      <td>−3.2%</td>
      <td>+2.0%</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>offload_wait</strong></td>
      <td rowspan="3"><strong>−37.4%</strong></td>
      <td rowspan="3"><strong>−26.9%</strong></td>
      <td rowspan="3"><strong>−13.9%</strong></td>
      <td><code>recsys.host.offload_wait</code></td>
      <td>−99.2%</td>
      <td>−98.9%</td>
      <td>−99.1%</td>
    </tr>
    <tr>
      <td><code>recsys.host.finish_task</code></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>recsys.gpu.release_offload_pages</code></td>
      <td>+53.4%</td>
      <td>+70.3%</td>
      <td>+119.7%</td>
    </tr>
    <tr>
      <td rowspan="9">onboard round (100% GPU miss, 100% CPU hit)</td>
      <td rowspan="5">lookup</td>
      <td rowspan="5">+18.7%</td>
      <td rowspan="5">+84.9%</td>
      <td rowspan="5">+101.6%</td>
      <td><code>recsys.gpu.lookup</code></td>
      <td>+37.5%</td>
      <td>+128.6%</td>
      <td>+142.9%</td>
    </tr>
    <tr>
      <td><code>recsys.host.build_index_meta</code></td>
      <td>+28.9%</td>
      <td>+87.7%</td>
      <td>+96.2%</td>
    </tr>
    <tr>
      <td><code>recsys.host.adapter.to_get_match_requests</code></td>
      <td>+61.3%</td>
      <td>+138.7%</td>
      <td>+162.1%</td>
    </tr>
    <tr>
      <td><code>flexkv.client.get_match</code></td>
      <td>+20.8%</td>
      <td>+94.9%</td>
      <td>+116.5%</td>
    </tr>
    <tr>
      <td><code>recsys.merge_lookup_results</code></td>
      <td>+0.5%</td>
      <td>+1.5%</td>
      <td>+2.0%</td>
    </tr>
    <tr>
      <td>allocate</td>
      <td>−3.6%</td>
      <td>−4.5%</td>
      <td>−5.9%</td>
      <td><code>recsys.gpu.allocate</code></td>
      <td>−3.4%</td>
      <td>−4.6%</td>
      <td>−6.4%</td>
    </tr>
    <tr>
      <td rowspan="2">onboard_launch</td>
      <td rowspan="2">−49.5%</td>
      <td rowspan="2">−49.6%</td>
      <td rowspan="2">−47.5%</td>
      <td><code>recsys.host._build_slot_mappings</code></td>
      <td>−73.3%</td>
      <td>−71.7%</td>
      <td>−69.6%</td>
    </tr>
    <tr>
      <td><code>flexkv.client.launch</code></td>
      <td>−1.9%</td>
      <td>−13.2%</td>
      <td>−6.2%</td>
    </tr>
    <tr>
      <td><strong>onboard_wait</strong></td>
      <td><strong>−19.3%</strong></td>
      <td><strong>−9.0%</strong></td>
      <td><strong>−19.9%</strong></td>
      <td><code>flexkv.client.wait</code></td>
      <td>−19.7%</td>
      <td>−9.1%</td>
      <td>−19.9%</td>
    </tr>
  </tbody>
</table>

#### 关键 op seq_len = 2048（L1, ms）

| op | ex1 Before (ms) | Optimized (ms) | Δ (%) |
| --- | --- | --- | --- |
| offload_launch | 7.192 | 3.345 | **−53.5%** |
| **offload_wait** | 30.655 | 22.422 | **−26.9%** |
| onboard_launch | 1.380 | 0.696 | **−49.6%** |
| **onboard_wait** | 21.244 | 19.341 | **−9.0%** |
| offload launch+wait | 37.847 | 25.767 | **−31.9%** |
| onboard launch+wait | 22.624 | 20.037 | **−11.4%** |
| **total** | 67.377 | 54.085 | **−19.7%** |

<!-- Donut 图（`seq_len=2048`）：`profiler_result_as_batch_880c660_rerun3/v2_slot_as_batch/run_1/plot/L1L2_latency_breakdown_latency_bs8_len2048.png`。 -->

### 1.4 six key APIs only

#### 绝对值 (ms)

<table>
  <thead>
    <tr>
      <th rowspan="2">mode</th>
      <th rowspan="2">L1 op</th>
      <th colspan="3">L1 (step-op, ms)</th>
      <th rowspan="2">L2 function</th>
      <th colspan="3">L2 (function, ms)</th>
    </tr>
    <tr>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">Optimized</td>
      <td rowspan="3">offload_launch</td>
      <td rowspan="3">2.647</td>
      <td rowspan="3">3.345</td>
      <td rowspan="3">4.203</td>
      <td><code>recsys.gpu.acquire_offload_pages</code></td>
      <td>0.091</td>
      <td>0.101</td>
      <td>0.112</td>
    </tr>
    <tr>
      <td><code>recsys.host._build_slot_mappings</code></td>
      <td>0.375</td>
      <td>0.425</td>
      <td>0.836</td>
    </tr>
    <tr>
      <td><code>put_match + batch launch</code> (unlabeled)</td>
      <td>2.182</td>
      <td>2.819</td>
      <td>3.255</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>offload_wait</strong></td>
      <td rowspan="3"><strong>15.981</strong></td>
      <td rowspan="3"><strong>22.422</strong></td>
      <td rowspan="3"><strong>34.374</strong></td>
      <td><code>recsys.host.offload_wait</code></td>
      <td>0.109</td>
      <td>0.161</td>
      <td>0.159</td>
    </tr>
    <tr>
      <td><code>recsys.host.finish_task</code></td>
      <td>15.688</td>
      <td>22.006</td>
      <td>33.939</td>
    </tr>
    <tr>
      <td><code>recsys.gpu.release_offload_pages</code></td>
      <td>0.089</td>
      <td>0.126</td>
      <td>0.134</td>
    </tr>
    <tr>
      <td rowspan="6">ex1 Before</td>
      <td rowspan="3">offload_launch</td>
      <td rowspan="3">6.682</td>
      <td rowspan="3">7.192</td>
      <td rowspan="3">7.948</td>
      <td><code>recsys.gpu.acquire_offload_pages</code></td>
      <td>0.098</td>
      <td>0.093</td>
      <td>0.102</td>
    </tr>
    <tr>
      <td><code>recsys.host._build_slot_mappings</code></td>
      <td>3.329</td>
      <td>3.303</td>
      <td>3.790</td>
    </tr>
    <tr>
      <td><code>flexkv.client.put_async</code></td>
      <td>2.529</td>
      <td>2.911</td>
      <td>3.192</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>offload_wait</strong></td>
      <td rowspan="3"><strong>25.514</strong></td>
      <td rowspan="3"><strong>30.655</strong></td>
      <td rowspan="3"><strong>39.922</strong></td>
      <td><code>recsys.host.offload_wait</code></td>
      <td>13.082</td>
      <td>15.180</td>
      <td>18.577</td>
    </tr>
    <tr>
      <td><code>recsys.host.finish_task</code></td>
      <td>0.010</td>
      <td>0.010</td>
      <td>0.011</td>
    </tr>
    <tr>
      <td><code>recsys.gpu.release_offload_pages</code></td>
      <td>0.058</td>
      <td>0.074</td>
      <td>0.061</td>
    </tr>
  </tbody>
</table>

#### Δ 相对 Before（inference total = `offload_launch` + `offload_wait`）

> **Δ = Optimized − Before**。下表分母均为各档 inference total（ms）；占比 Δ 单位为 **百分点 (pp)**。  
> inference total @2048：**37.85 → 25.77 ms（−31.9%）**；`offload_wait` L1 占比 **81.0% → 87.0%（+6.0 pp）** 是因为 launch 绝对降幅大于 wait，属占比重分配，不代表 wait 变慢。

##### Δ 绝对值 (ms)

<table>
  <thead>
    <tr>
      <th rowspan="2">L1 op</th>
      <th colspan="3">L1 Δ (ms)</th>
      <th rowspan="2">L2 function</th>
      <th colspan="3">L2 Δ (ms)</th>
    </tr>
    <tr>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">offload_launch</td>
      <td rowspan="3">−4.035</td>
      <td rowspan="3">−3.847</td>
      <td rowspan="3">−3.745</td>
      <td><code>recsys.gpu.acquire_offload_pages</code></td>
      <td>−0.007</td>
      <td>+0.008</td>
      <td>+0.010</td>
    </tr>
    <tr>
      <td><code>recsys.host._build_slot_mappings</code></td>
      <td>−2.954</td>
      <td>−2.878</td>
      <td>−2.954</td>
    </tr>
    <tr>
      <td><code>put_match + batch launch</code> (vs <code>put_async</code>)</td>
      <td>−0.347</td>
      <td>−0.092</td>
      <td>+0.063</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>offload_wait</strong></td>
      <td rowspan="3"><strong>−9.533</strong></td>
      <td rowspan="3"><strong>−8.233</strong></td>
      <td rowspan="3"><strong>−5.548</strong></td>
      <td><code>recsys.host.offload_wait</code></td>
      <td>−12.973</td>
      <td>−15.019</td>
      <td>−18.418</td>
    </tr>
    <tr>
      <td><code>recsys.host.finish_task</code></td>
      <td>+15.678</td>
      <td>+21.996</td>
      <td>+33.928</td>
    </tr>
    <tr>
      <td><code>recsys.gpu.release_offload_pages</code></td>
      <td>+0.031</td>
      <td>+0.052</td>
      <td>+0.073</td>
    </tr>
    <tr>
      <td><strong>inference total</strong></td>
      <td><strong>−13.568</strong></td>
      <td><strong>−12.080</strong></td>
      <td><strong>−9.293</strong></td>
      <td colspan="4"></td>
    </tr>
  </tbody>
</table>

##### Δ 占比 (pp)

<table>
  <thead>
    <tr>
      <th rowspan="2">L1 op</th>
      <th colspan="3">L1 Δ (pp)</th>
      <th rowspan="2">L2 function</th>
      <th colspan="3">L2 Δ (pp)</th>
    </tr>
    <tr>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
      <th>1024</th>
      <th>2048</th>
      <th>4096</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">offload_launch</td>
      <td rowspan="3">−6.54</td>
      <td rowspan="3">−6.02</td>
      <td rowspan="3">−5.71</td>
      <td><code>recsys.gpu.acquire_offload_pages</code></td>
      <td>+0.18</td>
      <td>+0.15</td>
      <td>+0.08</td>
    </tr>
    <tr>
      <td><code>recsys.host._build_slot_mappings</code></td>
      <td>−8.33</td>
      <td>−7.08</td>
      <td>−5.75</td>
    </tr>
    <tr>
      <td><code>put_match + batch launch</code> (vs <code>put_async</code>)</td>
      <td>+3.86</td>
      <td>+3.25</td>
      <td>+1.77</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>offload_wait</strong></td>
      <td rowspan="3"><strong>+6.54</strong></td>
      <td rowspan="3"><strong>+6.02</strong></td>
      <td rowspan="3"><strong>+5.71</strong></td>
      <td><code>recsys.host.offload_wait</code></td>
      <td>−40.05</td>
      <td>−39.48</td>
      <td>−38.40</td>
    </tr>
    <tr>
      <td><code>recsys.host.finish_task</code></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>recsys.gpu.release_offload_pages</code></td>
      <td>+0.30</td>
      <td>+0.29</td>
      <td>+0.22</td>
    </tr>
  </tbody>
</table>


## Micro-bench 2: Offload Stress — Effective KV Bandwidth

> **说明**：stress 测试无 ex1 Before 数据；本节对比 **v2（slot_mapping only）** vs **v2 + as_batch**，衡量 as_batch 在带宽侧的增量收益。

### 2.1 Configuration

| Parameter | Value |
| --- | --- |
| `num_layers` | **64** |
| `num_kv_heads` | **8** |
| `head_dim` | **256** |
| `dtype` | **bf16** |
| Peak ref BW (H100 D2H) | **64 GiB/s** |

### 2.2 bs=1, seq=1024（不触发 as_batch）

| launch_count | v2 eff. GiB/s | v2+as_batch eff. GiB/s | Δ |
| --- | --- | --- | --- |
| 50 | 44.4 | 44.3 | -0.3% |
| 100 | 44.4 | 45.0 | +1.4% |
| 150 | 46.1 | 45.5 | -1.2% |

### 2.3 bs=4, seq=512（as_batch 生效）

| launch_count | v2 eff. GiB/s | v2+as_batch eff. GiB/s | Δ | util vs 64 GiB/s |
| --- | --- | --- | --- | --- |
| 5 | 33.7 | 36.8 | +9.2% | 52.8% → 57.5% |
| 10 | 41.6 | 44.9 | +7.9% | 65.0% → 70.2% |
| 15 | 43.1 | 45.1 | +4.6% | 67.3% → 70.5% |
| 20 | 42.5 | 45.3 | +6.6% | 66.4% → 70.8% |

### 2.4 Conclusion

- **bs=1**：effective 带宽 **~44–46 GiB/s**，as_batch 无回归。
- **bs=4**：effective 带宽 **+5% ~ +9%**（33–43 → 37–45 GiB/s），峰值利用率 **~71%**。
