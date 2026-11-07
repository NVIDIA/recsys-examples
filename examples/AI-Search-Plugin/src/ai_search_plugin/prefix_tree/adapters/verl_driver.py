# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefix-tree training, verl driver side: keep GRPO siblings on one DP rank when balancing the batch.

verl's v1 trainer reorders every batch so DP ranks get similar token counts (``PPOTrainer._balance_batch``).  That
scatters the n samples of a prompt over ranks, and the tree packer in the training workers (``adapters.verl``) can then
only share prompts among whatever siblings happen to land together.  With ASRT_PREFIX_TREE=1 the balancer is replaced
by a group-level one: whole groups (key prefix = prompt uid, verl key format ``"{uid}_{session}_{index}"``) are
assigned to ranks, equal sample counts per rank, balancing *tree* tokens (prompt once + every response).
Falls back to verl's own balancer when the groups cannot be split evenly (counted).

Installed in the driver process by ``ai_search_plugin.prefix_tree.verl_plugin`` (post-import hook on
``verl.trainer.ppo.v1.trainer_base``).
"""
import os

_STATE = {'patched': False, 'calls': 0, 'grouped': 0, 'fallback': 0}
TRAINER_MODULE = 'verl.trainer.ppo.v1.trainer_base'


def enabled():
    return os.environ.get('ASRT_PREFIX_TREE', '0') not in ('0', '', 'false', 'False')


def group_key(k):
    """Prompt uid of a verl batch key "{uid}_{session}_{index}" (the key itself if it has another format)."""
    parts = str(k).rsplit('_', 2)
    return parts[0] if len(parts) == 3 else str(k)


def group_permutation(keys, tags, dp_size, partitioner):
    """Sample permutation that puts whole groups on DP ranks, or None when the groups cannot be split evenly.

    keys: batch keys; tags: per-sample dicts with 'seq_len' and 'prompt_len'; partitioner(workloads, k_partitions,
    equal_size) -> list of lists of group indices (verl's get_seqlen_balanced_partitions).
    Returns (perm, group parts as sample lists, tree tokens, flat tokens)."""
    groups, order = {}, []
    for i, k in enumerate(keys):
        g = group_key(k)
        if g not in groups:
            groups[g] = []
            order.append(g)
        groups[g].append(i)
    sizes = {len(v) for v in groups.values()}
    if len(order) % dp_size or len(sizes) != 1:
        return None
    tree = []
    for g in order:
        idx = groups[g]
        p = max(int(tags[idx[0]].get('prompt_len', 0)), 0)
        tree.append(p + sum(int(tags[i].get('seq_len', 0)) - p for i in idx))
    parts = partitioner(tree, k_partitions=dp_size, equal_size=True)
    perm = [i for part in parts for g in part for i in groups[order[g]]]
    assert sorted(perm) == list(range(len(keys)))
    sample_parts = [[i for g in part for i in groups[order[g]]] for part in parts]
    flat = sum(int(t.get('seq_len', 0)) for t in tags)
    return perm, sample_parts, sum(tree), flat


def _patch(mod):
    T = getattr(mod, 'PPOTrainer', None)
    if T is None or getattr(T._balance_batch, '_prefix_tree', False):
        return
    orig = T._balance_batch

    def _balance_batch(self, batch, metrics, logging_prefix='global_seqlen', keep_minibatch=False):
        _STATE['calls'] += 1
        if not enabled():
            return orig(self, batch, metrics, logging_prefix=logging_prefix, keep_minibatch=keep_minibatch)
        role, worker_group = 'actor', self.actor_rollout_wg
        if role not in worker_group._dispatch_info:
            worker_group._dispatch_info[role] = worker_group._query_dispatch_info(role)
        dp_size = max(worker_group._dispatch_info[role]) + 1
        batch = mod.upsample_batch_to_divisible_size(batch, self._get_required_batch_multiple(dp_size),
                                                     self.tokenizer.eos_token_id)
        res = group_permutation(batch.keys, batch.tags, dp_size, mod.get_seqlen_balanced_partitions)
        if res is None:
            _STATE['fallback'] += 1
            return orig(self, batch, metrics, logging_prefix=logging_prefix, keep_minibatch=keep_minibatch)
        perm, sample_parts, tree, flat = res
        seq = [int(t['seq_len']) for t in batch.tags]
        batch.reorder(perm)
        metrics.update(mod.log_seqlen_unbalance(seqlen_list=seq, partitions=sample_parts, prefix=logging_prefix))
        metrics[f'{logging_prefix}/prefix_tree_tokens_over_flat'] = float(tree) / max(1, flat)
        _STATE['grouped'] += 1
        return batch

    _balance_batch._prefix_tree = True
    T._balance_batch = _balance_batch
    _STATE['patched'] = True


def install():
    """Register the post-import patch of verl's v1 trainer (driver process).  Returns True when enabled."""
    if not enabled():
        return False
    from ...adapters._hooks import when_imported
    when_imported(TRAINER_MODULE, _patch)
    return True
