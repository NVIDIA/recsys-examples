# Experiments and ablations

English | [中文](EXPERIMENTS.zh-CN.md)

This is the long form of [README section 4](README.md#4-performance-speed-ups-and-where-time-goes): how we compared, every configuration, all health checks, the per-component ablations and what we did not measure.

> All numbers were measured with the original implementation this package was ported from. Current defaults differ slightly (for example `ws direct` is now off). Every number in sections 3–5 was independently reviewed (an independent reviewer recomputed it from the raw logs with their own scripts) unless the row says otherwise. Section 6 is **not reviewed and single-seed**.

Contents: [1. How we compare](#1-how-we-compare) · [2. Workloads and configurations](#2-workloads-and-configurations) · [3. Speed results](#3-speed-results) · [4. Where time goes](#4-where-time-goes) · [5. Component ablations](#5-component-ablations) · [6. Training quality (not reviewed)](#6-training-quality-not-reviewed-single-seed) · [7. Not measured](#7-not-measured) · [8. Known issues](#8-known-issues)

---

## 1. How we compare

| Item | Rule |
|---|---|
| Speed metric | **s/Mtok** = sum of step times in the window ÷ sum of tokens actually trained (prompt + response + tool output). We also report seconds per step where the two sides train different amounts per step. |
| Speed-up | baseline s/Mtok ÷ our s/Mtok. Above 1 = we are faster. "Time saved" = 1 − 1/speed-up. |
| Baseline | The fastest configuration of that framework we could find that keeps **every sample at most 2 policy versions stale** (staleness bound S = 2). The framework's own speed-ups (async modes, partial rollout, etc.) are on. Anything the baseline can get by changing its own config is given to the baseline. |
| Pairing | Each pair = one baseline run and one run of ours, back to back on the **same node**. Both orders (ours first / baseline first), balanced. Same model, data, hyper-parameters, seed, GPU count, training attention kernel (FA4 on both sides) and SGLang version. |
| Window | The first steps (warm-up, compilation) are dropped; the window is fixed before the run. |
| 95% CI | Bootstrap: inside each pair, resample the steps of each arm; combine pairs by geometric mean; 4,000–10,000 resamples. |
| Health checks (both sides) | Entropy; train–inference log-prob gap (ours ≤ 1.5× baseline, no rising trend); starting reward (steps 1–3 within 0.03); tokens trained per step (within ±5%); staleness ≤ 2 measured per token or per call, not only by the framework's own counter; retrieval latency probes (no failures, no spikes). |
| Ablation | Leave-one-out from our full set, same node, both orders, at least 2 pairs (4 when a single pair moves by more than ±5%). Ratio = s/Mtok without the component ÷ s/Mtok with everything. Above 1 = the component helps. Significant = the pooled 95% CI excludes 1 (or 0% for Qwen3-Next). |

---

## 2. Workloads and configurations

| Comparison | Model / task | Hardware | Algorithm, batch | Baseline (fastest compliant config) | Ours | Steps / window | Pairs (orders) |
|---|---|---|---|---|---|---|---|
| Search-R1 vs. slime | Qwen2.5-7B, multi-turn retrieval QA | 1× B300 (8 GPUs) | GRPO, 512 prompts × 5 | slime fully-async, 5 training + 3 rollout GPUs, selective recompute, concurrency 854, prefix-cache flush on every sync, 2 GB weight buffer, expandable segments, shared JIT cache. **Also uses our ticket admission** to stay within S = 2 (slime has no staleness cap of its own); an earlier test found slime with and without the cap within noise of each other | slime + full component set, same layout and concurrency | 3 warm-up + 25 timed; steps 3–24 | 2 (1 + 1) |
| Search-R1 vs. verl | same | 1× B300 | PPO (training includes the critic), 512 prompts × 1 | verl colocated async trainer (partial rollout, warm-up 1), `max_off_policy_threshold=3` (drops over-age samples, keeps S ≤ 2), bypass mode + token-level TIS, FSDP, SGLang `trtllm_mha`, 32 agent workers | verl + our verl components: `verl_ceiling` (S = 2, verl threshold 4 as a backstop), `verl_kvfresh`, fastload, `toolcache`, request routing | 20; steps 4–20 | 8 on 4 nodes (4 + 4) |
| ASearcher vs. verl | Qwen2.5-7B, multi-turn retrieval QA | 1× B300 | 128 prompts × 16 | verl colocated async, `max_off_policy_threshold=3` (drop), 4 GB weight buckets, 32 agent workers, FSDP; fastest of the verl layouts we tried | slime + full component set, concurrency 512, slime's `--use-distributed-post` on | 20; steps 2–19 (2–13 as sensitivity) | 4 (2 + 2) |
| ASearcher vs. slime (reference) | same | 1× B300 | same | slime fully-async, concurrency 256, `--use-distributed-post` on; 256 is the highest concurrency that stayed within S = 2 (288 exceeded it in 1 of 6 runs, 320 and above always) | same as above | 20; steps 2–19 | 4 (2 + 2) |
| Libra vs. slime | Qwen3-14B, Search-R1 task (short setting) | 1× B200 | GRPO, 32 prompts × 16 | slime best + **our ticket admission** (without it slime must drop to concurrency 8 to stay within S = 2), 2 GB weight buffer, expandable segments, shared JIT cache, prefix cache off (within noise of flush-on-sync, the faster one was kept) | slime + full component set (prefix cache on, kept correct by `kvfresh`) | 20; steps 2–19 | 6 (3 + 3) |
| Qwen3-Next vs. verl | Qwen3-Next 80B-A3B (MoE), multi-turn search with a simulated unstable search tool (scenario f in README section 2) | 4× B300: 2 training + 2 rollout nodes | 64 prompts × 8 | verl separate async, partial rollout, routing replay (R3), `max_off_policy_threshold=3` (drop); training TP2/PP2/CP2/EP8, full recompute; SGLang 8 × TP2, max running 256; NCCL weight sync, 4 GB buckets, multiple senders. 1 : 3 layout was 8.6% slower, so 1 : 1. Rejected by screening: other prefill-graph batch size, 8 GB buckets, 64 agent workers, 128 max running (all slower); selective recompute and MTP (out of memory); delta-sharded sync (broke training) | slime + ticket S = 2 + DEFER, concurrency 256, full component set incl. expert placement, `response_logits`, prefetch; MTP off; `ws direct` **on** | 10; steps 3–10 | 4 (2 + 2) |

Notes
- Retrieval: the small models use a local retriever. Every pair checked retrieval latency with probes on both arms; pairs with retrieval incidents were excluded or marked.
- Qwen3-Next: verl needed three local fixes before it was usable (see 3.2); both arms used the same SGLang build. The two arms used the default NCCL of their own images (2.30.7 vs. 2.29.7).

---

## 3. Speed results

### 3.1 Summary

| Comparison | Speed-up [95% CI] | Time saved for the same training | Baseline / ours s/Mtok | Per-pair speed-ups | Verdict |
|---|---|---|---|---|---|
| Search-R1 vs. slime | **1.170×** [1.164, 1.177] | ~15% | 18.31 / 15.64 | 1.171, 1.169 | All checks pass |
| Search-R1 vs. verl | **1.002×** [0.984, 1.021] | tie | 47.96 / 47.87 | 1.079, 0.982, 0.966, 0.994, 1.014, 0.993, 0.987, 1.001 | **Tie**, two checks fail (3.2) |
| ASearcher vs. verl | **1.167×** [1.131, 1.208] | ~14% | 16.52 / 14.36 | 1.140, 1.168, 1.193, 1.167 | With caveats (3.2) |
| ASearcher vs. slime (reference) | 1.299× [1.235, 1.355] | ~23% | 18.97 / 14.70 | 1.233, 1.330, 1.301, 1.333 | Reference only |
| Libra vs. slime | **1.219×** [1.133, 1.310] | ~18% | 45.81 / 37.51 | 1.150, 1.247, 1.196, 1.308, 1.172, 1.251 | All checks pass; baseline uses our ticket |
| Qwen3-Next vs. verl | **1.560×** [1.389, 1.719] | ~36% | 34.22 / 21.92 (mean of pairs) | 1.541, 1.491, 1.642, 1.570 | With 6 caveats (3.2) |

The s/Mtok columns for the small models are pooled over all pairs (sum of time ÷ sum of tokens); the speed-up is the geometric mean of per-pair ratios, so the two can differ slightly (ASearcher vs. verl: pooled 1.150, per-pair 1.167, because the pair with the longest responses weighs more when pooled).

### 3.2 Per comparison: health checks and caveats

**Search-R1 vs. slime — 1.170×**
- By step: 1.176× [1.166, 1.187]. Tokens per step differ by ≤ 0.8%.
- Health checks, all pass: entropy 1.63–1.77 on both sides over the first 3 steps; starting reward 0.13–0.15 on both; log-prob gap (slime's own metric) 0.0187–0.0189 on both, flat; staleness worst 2 on both; mean response length 443–447 on both; 0 failed retrieval probes.

**Search-R1 vs. verl — 1.002×, tie**
- By step we are **4.6% slower** (0.956× [0.928, 0.985]), because we train more per step: 4.8% more tokens in total (8 pairs pooled) and more model-written tokens (+6.8% on the first 4 pairs, more on the rest).
- Pass: entropy (−5.0% to +2.1%), log-prob gap (0.86–1.14×), first-step reward (≤ 0.027 apart).
- **Fail, two checks**: tokens per step differ by more than ±5% in 4 of 8 pairs (+5.2%, +10.5%, +7.3%, +7.1%); steps 1–3 reward differs by −0.035 in one pair (threshold 0.03).
- Likely reason (consistent, **not proven**): the baseline drops over-age samples (264–687 per run), we drop none. One untimed baseline run that happened to drop nothing trained about as many loss tokens per step as our runs (not reviewed).
- `toolcache` did not activate here (a positional-argument call bypassed the cache), and `tokcache`/`route` had nothing to do (n = 1 per prompt). So this tie says nothing about those components.

**ASearcher vs. verl — 1.167×**
- Sensitivity, steps 2–13: 1.251× [1.213, 1.292]. In one pair the verl arm's responses grew sharply after step 15 (2.2× our tokens per step), which makes verl cheaper per token; without that pair: 1.176×. Both directions favor verl, so the main number is conservative.
- Ours: staleness worst 2 (per call), log-prob gap matches what its staleness mix predicts (measured / predicted 0.92–0.99), all pass.
- verl: **staleness compliance not proven**. Its own label says worst 2; our per-segment log gives an upper bound of 3 for 0.002–0.16% of trajectories. If verl did exceed the bound, it gained speed from it, so this favors verl. verl's log-prob gap was not judged (only trajectory-level staleness is available).
- Other checks: entropy and reward curves of the same shape (start 0.12–0.13 ours / 0.15 verl); both train 128 × 16 per step; FA4 on both.

**ASearcher vs. slime — 1.299×, reference only**
- slime's concurrency 256 is compliant (worst 2). One of the four pairs failed the late log-prob-rise check (1.59 vs. threshold 1.5, our arm; not reproduced in a rerun with the same order). Without that pair: 1.298× [1.220, 1.364]. Reported for reference because the main ASearcher baseline is verl.

**Libra vs. slime — 1.219×**
- Health checks, all pass: 20 steps × 512 samples on every run; entropy 0.27–0.31 on both; log-prob gap 0.0131–0.0143 on both, flat; starting reward 0.27–0.37; staleness worst 2 on both; FA4 on both.
- Order effect: ours-first pairs 1.150 / 1.196 / 1.172, baseline-first 1.247 / 1.308 / 1.251. The 3 + 3 balance cancels it.
- The baseline uses our ticket admission. Without it, slime has to drop to concurrency 8 to stay within S = 2; ticket vs. that setting is 3.01× (section 5.4, not counted in the 1.219×).

**Qwen3-Next vs. verl — 1.560×**
- By step: 1.548×. Order: ours first 1.530, verl first 1.591 (intervals overlap).
- Six pairs were started; one hung on each side (verl in an EP all-gather, ours in a weight-sync broadcast), so one pair per side was dropped; 2 + 2 orders remain.
- Health checks, 9 items × 4 pairs × both arms, all pass: tokens per step within 1.37%; first-step reward within 0.009; staleness worst 2 (ours) / 1–2 (verl); FA2 fallback 0 on both; prefix-cache flush failures 0; search-proxy latency identical (p50 0.317 / 0.316 s, p95 1.81 / 1.81 s).
- Accuracy check before timing (10 steps each arm):

| | Ours | verl after fixes | verl before fixes |
|---|---|---|---|
| Train–inference log-prob gap | 0.0138–0.0142, mean **0.0139**, flat | 0.016 (even steps) – 0.021 (odd), mean **0.0178** | **0.210** |
| Entropy | 0.754–0.816 | 0.77–0.81 | 0.63–0.67 |
| Reward (simulated search, trend only) | 0.480–0.508 | 0.49–0.51 | — |
| Staleness worst | 2 | 2 | — |

- Caveats:
  1. **One scenario only**: simulated search (slow at times, timeouts, errors), so rewards are a trend only. Scaled-down layout: 2 training + 2 rollout nodes, 64 × 8.
  2. **`ws direct` was on** during the measurement. It was later turned off by default after rare hangs (1 in 60 syncs; verl also hung 1 in 63, cause unknown) and not re-measured. The whole weight-sync group is worth ~1%, not significant.
  3. **Conservative**: ticket admission was on; turning it off here is 2.3% faster and still compliant (section 5.5).
  4. **Staleness is used differently**: our samples were almost all 2 steps stale, verl's ~99.6% only 1 step; both within the bound. Part of the gain may come from this; not measured separately.
  5. **Not attributable to components yet**: only prefetch (+13.5%) and expert placement (+2.7%) are significant in the ablations. Do not add or subtract the per-stage numbers of section 4 and the ablations.
  6. **verl needed three local fixes**, not upstreamed: a newer SGLang (0.5.21) first on the path (the older one reported decode log-probs systematically off from prefill on multi-turn sequences), an adapter for it, and the original GDN output-norm weights (the converter stored them in a lossy form). Both arms used the same SGLang.

---

## 4. Where time goes

Per-stage time in s/Mtok (small models) and share of the step. "Waiting" = waiting for rollout data, excluding weight sync. Small models: independently reviewed. Qwen3-Next: s/step, **not reviewed**.

| Comparison | Arm | Training | Waiting | Sync | Other | Step total |
|---|---|---|---|---|---|---|
| Search-R1 vs. slime | Baseline | 15.67 (86%) | 2.57 (14%) | 0.07 (0%) | 0 | 18.31 |
| | Ours | 13.93 (89%) | 1.59 (10%) | 0.12 (1%) | 0 | 15.64 |
| Search-R1 vs. verl (PPO, training incl. critic) | Baseline | 31.52 (66%) | 8.19 (17%) | 6.63 (14%) | 1.62 (3%) | 47.96 |
| | Ours | 30.53 (64%) | 9.10 (19%) | 6.67 (14%) | 1.57 (3%) | 47.87 |
| ASearcher vs. verl | Baseline | 12.13 (73%) | 3.64 (22%) | 0.66 (4%) | 0.09 (1%) | 16.52 |
| | Ours | 12.61 (88%) | 1.66 (12%) | 0.08 (1%) | 0 | 14.36 |
| ASearcher vs. slime (reference) | Baseline | 13.47 (71%) | 5.43 (29%) | 0.07 (0%) | 0 | 18.97 |
| | Ours | 12.67 (86%) | 1.96 (13%) | 0.07 (0%) | 0 | 14.70 |
| Libra vs. slime | Baseline | 30.14 (66%) | 14.92 (33%) | 0.75 (2%) | 0 | 45.81 |
| | Ours | 28.51 (76%) | 8.03 (21%) | 0.97 (3%) | 0 | 37.51 |
| Qwen3-Next vs. verl (s/step, not reviewed) | Baseline | 211.3 (60%) | 102.6 (29%) | 37.5 (11%) | 1.3 (0%) | 352.96 |
| | Ours | 168.7 (74%) | 51.9 (23%) | 6.9 (3%) | 0 | 227.85 |

How the gap splits (baseline − ours):
- Search-R1 vs. slime: 2.67 s/Mtok; 65% in training (response-only compute), 37% in waiting.
- Search-R1 vs. verl: almost none; we are 0.99 s/Mtok faster in training and 0.91 s/Mtok slower in generation, hence the tie.
- ASearcher vs. slime: 4.27 s/Mtok; 81% in waiting. Libra: 8.30 s/Mtok; 83% in waiting.
- ASearcher vs. verl: we are slower in training per token (verl's long-response pair makes its training cheap per token) and faster in generation and sync.
- Qwen3-Next: training and waiting are each ~43 s/step shorter. Our sync (6.9 s) overlaps with waiting; verl's 37.5 s is added to every step. verl's generation varies widely: 0–13 s on short steps, 194–236 s on long ones (not reviewed).
- Retrieved text is 52% of all tokens on Search-R1 and 70–83% on ASearcher.

Two more observations, valid only for these workloads:
- **Ticket admission helps only where a plain concurrency cap breaks the bound**: needed on Libra and ASearcher; no effect on Search-R1 (×0.996); 2.3% slower on Qwen3-Next.
- **Faster sync doesn't always matter**: on Qwen3-Next, disabling engine hooks doubled sync time but step time did not change, because sync runs while waiting for data (not reviewed).

---

## 5. Component ablations

Ratio = s/Mtok without the component ÷ s/Mtok with the full set; above 1 = helps. ✓ = significant (pooled 95% CI excludes 1). "Not active" = the component never did anything in that configuration, so no conclusion about it. Each task has a table of independently reviewed rows and, below it, the rows that were not (or only partly) reviewed.

### 5.1 Search-R1 · slime (2 pairs per item, both orders, steps 3–24)

| Component | What was removed | Ratio [95% CI] | Sig. | Baseline can get it? | Review |
|---|---|---|---|---|---|
| `response_logits` | Output layer only on response tokens | **1.116** [1.109, 1.122] | ✓ | No | Ratio recomputed |
| `ticket_admission` | Ticket admission (both arms at concurrency 854) | 0.996 [0.990, 1.001] | — | No | Ratio recomputed; the no-ticket arm's staleness (worst 2, rebuilt from logs) not reviewed |
| `weight_sync` | flatten + parallel PP send (2 GB buffer kept on both) | 1.002 [0.996, 1.008] | — | 2 GB buffer yes (both arms have it); rest no | Ratio recomputed |

Partly reviewed (health checks reviewed, ratio not recomputed):

| Component | What was removed | Ratio [95% CI] | Sig. |
|---|---|---|---|
| `prefetch` | Fetch next batch during weight sync | 1.033 [1.027, 1.038] | ✓ |
| `sampler_nosync` + fastload | Engine hooks (plus two helper hooks) | 1.002 [0.996, 1.008] | — |

Reading: the gain comes mostly from `response_logits` and `prefetch`; their product is 1.152 against the measured 1.170. Leave-one-out ratios do not multiply exactly, and about 1.5% is not explained by any single item; the cause was not measured (a cumulative build-up table was not run).

### 5.2 Search-R1 · verl (steps 4–20)

| Component | Ratio [95% CI] (pairs) | Sig. | Note | Review |
|---|---|---|---|---|
| fastload (engine hook) | 0.986 [0.962, 1.011] (2) | — | | Recomputed, health checks pass |
| `verl_kvfresh` (cost of the guard) | 0.993 [0.968, 1.016] (2) | — | Cost ~0.7%, not significant | Recomputed, health checks pass |
| route + balance | 0.998 [0.971, 1.025] (2) | — | Almost idle by design here (n = 1 per prompt) | Recomputed, health checks pass |
| `toolcache` | 1.011 [0.984, 1.039] (2) | — | **Not active**: cache bypassed by a positional-argument call | Recomputed, health checks pass |
| tokcache | 1.006 [0.986, 1.027] (4) | — | **Not active** (0 hits); tokens-per-step check failed in 3 of 4 pairs, no conclusion | Recomputed |

Reference only: `verl_ceiling` + off-policy threshold 1.000 [0.977, 1.024]. The ceiling never triggered, so this really measured verl's own threshold 4 → 3, and the tokens-per-step check failed. Not counted as our gain.

Reading: 1.002× overall, no attributable gain in this configuration. Several components were not active here, so this does not show they are useless.

### 5.3 ASearcher (vs. our full set on slime; steps 2–19)

| Component | Ratio [95% CI] (pairs) | Sig. | slime baseline can get it? | verl has it? | Review / health |
|---|---|---|---|---|---|
| DEFER (move slow groups to a later batch) | **1.333** [1.252, 1.412] (4) | ✓ | No | Partly (verl drops instead) | Reviewer recomputed the first 2 pairs: 1.286 [1.162, 1.410] ✓. 1 of 4 pairs failed the late log-prob-rise check (1.64, caused by the no-DEFER arm's late decline) |
| `sampler_nosync` + fastload | **1.116** [1.027, 1.196] (4) | ✓ | No | No | Reviewer recomputed the first 2 pairs: 1.171 [1.036, 1.283] ✓ |
| `response_logits` | **1.090** [1.043, 1.135] (4) | ✓ | No | No | Recomputed; 1 of 4 pairs failed the late-rise check (1.52) |
| `rshard` | 1.051 [0.995, 1.112] (4) | — | No (slime's `--use-distributed-post` is on in both arms) | Partly | Recomputed; 1 of 4 pairs failed the late-rise check (1.87) |
| `ticket_admission` | No compliant comparison | — | No | Partly | Without ticket, concurrency 256 / 192 / 128 / 64 all exceeded S = 2. For reference only: at 256, 0.943 [0.908, 0.975] (2 pairs); at 64, 1.041 [0.907, 1.157] (4 pairs, one outlier pair with an abnormal log-prob gap). **This does not mean ticket has no gain.** |

Not reviewed (ratio not recomputed):

| Component | Ratio [95% CI] (pairs) | Sig. |
|---|---|---|
| `prefetch` | 1.066 [1.012, 1.130] (2) | weak ✓ |
| `toolcache` + `tool_hedge` | 1.079 [0.974, 1.180] (4) | — |
| `weight_sync` group (incl. `ws direct`) | 0.999 [0.945, 1.058] (4) | — |

These three stay on in the default set (no measurable gain with the current pairs; not re-tested).

Note on ASearcher health checks: two per-pair checks (tokens per step within ±5%, early log-prob gap ≤ 1.1×) fail even between two runs of the same configuration in this 20-step window, so they cannot separate component effects here; rows where they fail are kept and marked. Speed ratios are per trained token, so they are not affected.

### 5.4 Libra (4 pairs per item, both orders, steps 2–19)

| Component | Ratio [95% CI] | Sig. | Baseline can get it? | Review |
|---|---|---|---|---|
| `sampler_nosync` + fastload (engine hooks) | **1.151** [1.045, 1.267] | ✓ | No | Recomputed |
| `kvfresh` + prefix cache on, vs. prefix cache off | 1.065 [0.967, 1.171] | — | No (cache off is what the baseline uses) | Recomputed. A real speed difference, included in the 1.219× |

Cumulative build-up (reviewed): adding components one at a time from the baseline (1 pair × 2 orders per level, so no per-level conclusion) multiplies to **1.226**, matching 1.219×. Most of it comes from +engine hooks (1.100) and +kvfresh with prefix cache (1.110); the other levels are 0.979–1.046.

Not reviewed (ratio not recomputed):

| Component | Ratio [95% CI] | Sig. |
|---|---|---|
| `sampler_nosync` alone | 1.064 [0.963, 1.172] | — |
| fastload alone | 1.010 [0.932, 1.092] | — |
| `response_logits` | 1.053 [0.980, 1.127] | — |
| `toolcache` + `tool_hedge` | 1.032 [0.950, 1.121] | — |
| `weight_sync` (flatten + parallel send) | 1.030 [0.946, 1.124] | — |
| `rshard` | 1.029 [0.940, 1.123] | — |
| `prefetch` | 0.989 [0.923, 1.060] | — |
| `kvfresh` vs. slime's native flush-on-sync (2 pairs) | 1.019 [0.914, 1.136] | — |
| `ticket_admission` vs. slime's only compliant setting (no ticket, concurrency 8; 2 pairs) | 3.01 [2.77, 3.27] | ✓ (not counted in 1.219× because the baseline borrows ticket) |

Reading: only the engine hooks are significant alone; nosync and fastload are not significant separately. The 1.219× comes from the combination.

### 5.5 Qwen3-Next (1 pair per order, 10 steps per segment, steps 3–10)

Gain = s/Mtok without ÷ s/Mtok with − 1; positive = the component helps. Significance by the pooled two-order CI, a rule fixed before the results.

| Component | Gain, pooled [95% CI] | Per order | Sig. | Where the time moved (s/step, not reviewed) |
|---|---|---|---|---|
| `prefetch` (+ `r3_transfer`) | **+13.51%** [+11.56, +15.59] | +13.08% / +13.93% | ✓ | Waiting +30 s/step without it; where exactly is not yet known |
| `expert_placement` | **+2.70%** [+1.21, +4.38] | +2.29% / +3.10% | ✓ | Training ~4 s/step |
| `response_logits` | +1.01% [−0.82, +2.99] | +0.59% / +1.42% | — | |
| DEFER (with the straggler cut) | +0.38% [−1.20, +2.00] | −1.84% / +2.65% | — | |
| `sampler_nosync` + fastload | −0.29% [−2.39, +1.76] | +0.12% / −0.69% | — | Sync time doubles without it, step time unchanged (sync overlaps waiting) |
| `ticket_admission` | **−2.32%** [−3.92, −0.60] | −3.41% / −1.23% | ✓ negative | Off is faster and still compliant (worst 2, plain group-count cap); kept on, so 1.560× is conservative |

Not reviewed:

| Component | Result | Note |
|---|---|---|
| `weight_sync` group (incl. `ws direct`) | +1.2% [−0.3, +2.7], one order only | The other order hung twice (`ws direct`), so not judged |
| `r3pack` | −1.89% / +1.21% | Never actually packed (turned itself off); this is effectively an A/A noise check |
| MTP speculative decoding (stub here) | −3.3% and −1.6% (two different setups) | Slower both times; kept off; reason unknown |
| r3keep | Never triggered | |
| GDN/FLA kernels, routing replay itself | Same on both arms | Not counted as our gain |

Noise reference (not reviewed): our full arm on 8 different node blocks ranged 21.62–22.29 s/Mtok (3.1%).

---

## 6. Training quality (not reviewed, single seed)

These are **not independently reviewed** and each arm is **one training seed**. The confidence intervals cover evaluation sampling only, not seed-to-seed variation (estimated at .01–.03 EM).

**Search-R1 · verl, 500 steps, exact-match (EM) averaged over 7 test sets (51,713 questions)**

| Step | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| Baseline (verl best) | .318 | .398 | .427 | .437 | .444 |
| Ours (default config) | .338 | .409 | .438 | .442 | .440 |

- Step 500: ours − baseline = −.004 [−.015, +.008], **level**. Step 300: +.011 (CI excludes 0).
- Health checks on both: starting reward .12–.13, entropy falls steadily, log-prob gap ≤ .006 without a rising trend, staleness worst 2.
- A separate non-default arm with an extra response-only loss path on verl ended .024 lower at step 500; the cause was not found, and that path is not part of the default config.

**Libra, 200 steps, training reward**
- Steps to reach a 10-step moving average of 0.40: baseline step 49, ours step 47. Level, within noise.

---

## 7. Not measured

- **Paired comparisons against the original implementations, AReaL and ROLL.** AReaL and ROLL were only timed on their own; they were not compared with us on the same node in pairs, so they are not reported here.
- **Accuracy** for ASearcher, Search-R1 · slime and Qwen3-Next (its search tool is simulated, so its reward is a trend only).
- Qwen3-Next: real retrieval; other scenarios; slime vs. verl as the stronger baseline; sparse-attention models; `ws direct` on its own; numbers with `ws direct` off.
- Multi-node runs for the small models; the prefill vs. decode split.
- Components not in any measured configuration: `prefix_tree`, `r3_ep_plan`, `hybridep_overlap`. Not ablated: `pool_fill`, `jitfix`, `jitcache`.
- A cumulative build-up table for Search-R1 · slime (to explain the remaining ~1.5%).

---

## 8. Known issues

| Issue | Status |
|---|---|
| `ws direct` hung weight sync on the 80B model (1 in 60 syncs; one rollout engine never joined a broadcast) | Off by default; root cause not confirmed. 391 small-model runs (7,642 syncs) had 0 hangs, but this is not extrapolated |
| verl also hung on Qwen3-Next (1 in 63 syncs) | Cause not confirmed |
| verl on Qwen3-Next needed three local fixes | Not upstreamed |
| Search-R1 vs. verl: two health checks fail | Linked to the baseline dropping over-age samples; not proven |
| ASearcher vs. verl: verl's staleness compliance not proven | Favors verl if anything |
| `toolcache` bypassed in the Search-R1 · verl adapter (positional-argument call) | Not fixed in the measured runs |
| `ticket_admission` is 2.3% slower than a plain cap on Qwen3-Next and has no effect on Search-R1 · slime (×0.996) | It helps only where a plain concurrency cap cannot hold the bound (Libra, ASearcher) |
| Unexplained: where prefetch's 30 s comes from on Qwen3-Next; why MTP is slower; why verl's odd steps have a higher log-prob gap; why r3keep triggers when engine hooks or ticket are off; ~1.5% of Search-R1 · slime | Open |
