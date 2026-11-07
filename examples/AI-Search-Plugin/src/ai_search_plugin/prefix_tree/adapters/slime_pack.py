# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sibling-aware micro-batch packing for slime (rollout-manager side), companion of ``adapters.slime``.

slime's ``build_dp_schedule`` packs each step's samples into micro-batches by length only (first fit in rollout
order, then deals micro-batches to DP ranks).  The tree pack (``adapters.slime``) can only share a prompt among the
siblings that land in the same micro-batch, and first fit often scatters a GRPO group over several micro-batches.
This packer keeps slime's contract (same steps, same samples per step, the same number of micro-batches on every DP
rank, local index ranges) but fills micro-batches with *sibling units* (a group's trunk once + its branches) counted
in tree tokens against the same ``max_tokens_per_gpu`` cap; a group larger than the cap is split (its trunk repeated).
Groups use ``plan.group_by_prompt`` (identical prompt tokens; trunk = longest common prefix), i.e. exactly the groups
the training side builds.

Use: ``--rollout-function-path ai_search_plugin.prefix_tree.adapters.slime_pack.generate_rollout_fully_async`` with
PREFIX_TREE_ROLLOUT_INNER = the rollout function it wraps (default: slime's stock fully-async rollout),
ASRT_PREFIX_TREE=1 (PREFIX_TREE_PACK=0 turns only the packer off).  Applies with dynamic batch size, CP 1, no VPP and
when the manager holds the tokens; anything else keeps slime's schedule (logged).
"""
import json
import os
import time

import numpy as np

from ..plan import group_by_prompt

_MGR = {'installed': False, 'data': None, 'n': 0}
STOCK_ROLLOUT = 'slime.rollout.fully_async_rollout.generate_rollout_fully_async'


def enabled():
    return (os.environ.get('ASRT_PREFIX_TREE', '0') not in ('0', '', 'false', 'False')
            and os.environ.get('PREFIX_TREE_PACK', '1') == '1')


def _min_trunk():
    return int(os.environ.get('PREFIX_TREE_MIN_TRUNK', '16'))


def _units(idx, tokens, resp, cap):
    """Sibling units of one step: [{'T', 'loc' (positions into idx), 'size' (tree tokens)}], plus sample lengths."""
    seqs, plens, lens = [], [], []
    for i in idx:
        t = tokens[i]
        a = np.asarray(t.cpu().numpy() if hasattr(t, 'cpu') else t, dtype=np.int64)
        seqs.append(a)
        lens.append(len(a))
        plens.append(len(a) - int(resp[i]))
    units = []
    for T, loc in group_by_prompt(seqs, plens, _min_trunk()):
        if not T:
            units += [{'T': 0, 'loc': [j], 'size': lens[j]} for j in loc]
            continue
        cur, size = [], T
        for j in loc:
            add = lens[j] - T
            if cur and size + add > cap:
                units.append({'T': T, 'loc': cur, 'size': size} if len(cur) > 1 else
                             {'T': 0, 'loc': cur, 'size': lens[cur[0]]})
                cur, size = [], T
            cur.append(j)
            size += add
        units.append({'T': T, 'loc': cur, 'size': size} if len(cur) > 1 else
                     {'T': 0, 'loc': cur, 'size': lens[cur[0]]})
    return units, lens


def _split_unit(u, lens):
    loc = u['loc']
    half = len(loc) // 2
    out = []
    for part in (loc[:half], loc[half:]):
        if len(part) == 1:
            out.append({'T': 0, 'loc': part, 'size': lens[part[0]]})
        else:
            out.append({'T': u['T'], 'loc': part, 'size': u['T'] + sum(lens[j] - u['T'] for j in part)})
    return out


def tree_schedule(args, tpc, total_lengths, global_batch_size, rollout_indices, data, workloads=None,
                  balanced_partitions=None):
    """Same contract as slime.utils.dp_schedule.build_dp_schedule.

    Returns ((partitions, micro_batch_indices, num_microbatches, global_batch_sizes), per-step stats), or
    (None, reason) when it does not apply."""
    dp, cp = tpc['dp_size'], tpc['cp_size']
    if cp != 1 or tpc.get('vpp_size', 1) > 1 or not args.use_dynamic_batch_size:
        return None, 'cp/vpp/static batch'
    cap = args.max_tokens_per_gpu * cp
    tokens, resp = data.get('tokens'), data.get('response_lengths')
    if tokens is None or resp is None:
        return None, 'no tokens in the manager'
    rid_to = {}
    for pos, rid in enumerate(rollout_indices):
        rid_to.setdefault(rid, []).append(pos)
    rids = list(rid_to)
    steps = len(rids) // global_batch_size
    partitions = [[] for _ in range(dp)]
    mbi = [[] for _ in range(dp)]
    nmb, gbs, stats = [], [], []
    for s in range(steps):
        idx = [p for rid in rids[s * global_batch_size:(s + 1) * global_batch_size] for p in rid_to[rid]]
        units, lens = _units(idx, tokens, resp, cap)
        bins, sums = [], []
        for k, u in enumerate(units):          # first fit, units in rollout order (as slime's first-fit pack)
            for j in range(len(bins)):
                if sums[j] + u['size'] <= cap:
                    bins[j].append(k)
                    sums[j] += u['size']
                    break
            else:
                bins.append([k])
                sums.append(u['size'])
        target = max(-(-len(bins) // dp) * dp, dp)
        while len(bins) < target:
            cand = [j for j in range(len(bins)) if len(bins[j]) > 1]
            if cand:   # split the largest multi-unit bin in two (units stay whole)
                j = max(cand, key=lambda x: sums[x])
                left, right, sl, sr = [], [], 0, 0
                for k in sorted(bins[j], key=lambda k: -units[k]['size']):
                    if sl <= sr:
                        left.append(k)
                        sl += units[k]['size']
                    else:
                        right.append(k)
                        sr += units[k]['size']
                bins[j], sums[j] = left, sl
                bins.append(right)
                sums.append(sr)
                continue
            cu = [k for k in range(len(units)) if len(units[k]['loc']) > 1]
            if not cu:
                return None, f'step {s}: cannot reach {target} micro-batches'
            k = max(cu, key=lambda x: units[x]['size'])
            a_u, b_u = _split_unit(units[k], lens)
            jb = next(j for j in range(len(bins)) if k in bins[j])
            sums[jb] += a_u['size'] - units[k]['size']
            units[k] = a_u
            units.append(b_u)
            bins.append([len(units) - 1])
            sums.append(b_u['size'])
        K = len(bins)
        nmb.append(K // dp)
        gbs.append(global_batch_size)
        if getattr(args, 'balance_data', False) and balanced_partitions is not None:
            w = workloads(sums, args) if workloads is not None else list(sums)
            rank_bins = balanced_partitions(w, dp, equal_size=True)
        else:
            rank_bins = [list(range(r, K, dp)) for r in range(dp)]
        for r in range(dp):
            for j in rank_bins[r]:
                loc = [l for k in bins[j] for l in units[k]['loc']]
                start = len(partitions[r])
                partitions[r].extend(idx[l] for l in loc)
                mbi[r].append(list(range(start, start + len(loc))))
        stats.append({'step': s, 'samples': len(idx), 'units': len(units),
                      'sib_units': sum(1 for u in units if u['T']), 'mbs': K, 'tree_tokens': int(sum(sums)),
                      'flat_tokens': int(sum(lens)), 'max_mb_tree': int(max(sums)), 'cap': int(cap)})
    return (partitions, mbi, nmb, gbs), stats


def _log(rec):
    d = os.environ.get('ASRT_LOG_DIR')
    if d:
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, 'prefix_tree_pack.jsonl'), 'a') as f:
                f.write(json.dumps(rec) + '\n')
        except OSError:
            pass
    print(f'[asrt prefix_tree.pack] {json.dumps(rec)[:600]}', flush=True)


def install():
    """Wrap slime's batch builder (rollout manager).  Idempotent."""
    if _MGR['installed']:
        return True
    import slime.data.batch_builder as bb
    from slime.utils.dp_schedule import _calculate_workloads
    from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
    orig_build = bb.build_dp_schedule
    orig_split = bb.BatchBuilder.split_by_dp

    def split_by_dp(self, data, *a, **k):
        _MGR['data'] = data if isinstance(data, dict) else None
        try:
            return orig_split(self, data, *a, **k)
        finally:
            _MGR['data'] = None

    def build_dp_schedule(args, tpc, total_lengths, *, global_batch_size, rollout_indices):
        res = orig_build(args, tpc, total_lengths, global_batch_size=global_batch_size,
                         rollout_indices=rollout_indices)
        if not enabled() or _MGR['data'] is None:
            return res
        t0 = time.perf_counter()
        try:
            out, info = tree_schedule(args, tpc, total_lengths, global_batch_size, rollout_indices, _MGR['data'],
                                      workloads=lambda s, a: _calculate_workloads(list(s), a),
                                      balanced_partitions=get_seqlen_balanced_partitions)
        except Exception as e:  # noqa: BLE001  never break training: keep slime's schedule
            out, info = None, f'error {e!r}'[:300]
        _MGR['n'] += 1
        if out is None:
            _log({'n': _MGR['n'], 'applied': False, 'why': info})
            return res
        _log({'n': _MGR['n'], 'applied': True, 's': round(time.perf_counter() - t0, 3), 'slime_mbs': res[2],
              'tree_mbs': out[2], 'steps': info})
        return out

    bb.build_dp_schedule = build_dp_schedule
    bb.BatchBuilder.split_by_dp = split_by_dp
    _MGR['installed'] = True
    return True


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """``--rollout-function-path`` wrapper: installs the packer, then runs PREFIX_TREE_ROLLOUT_INNER."""
    import importlib
    if enabled():
        install()
    inner = os.environ.get('PREFIX_TREE_ROLLOUT_INNER', STOCK_ROLLOUT)
    mod, fn = inner.rsplit('.', 1)
    return getattr(importlib.import_module(mod), fn)(args, rollout_id, data_buffer, evaluation=evaluation)
