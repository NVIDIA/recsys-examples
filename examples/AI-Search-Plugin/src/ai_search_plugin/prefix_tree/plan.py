# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefix-tree packing of GRPO groups: host-side layout, index maps and sibling grouping (numpy only).

One group = n samples that share a token prefix (GRPO siblings share the prompt).  The tree layout puts, per group,

  trunk   : the shared prefix ``tokens[:T]`` once (T a multiple of the THD alignment, T <= shared length - 1, so the
            last shared token - whose output predicts the first token that differs - stays in every branch);
  branches: for each sample its remaining tokens ``tokens[T:]``, zero-padded at the end to the alignment.

Branch position ids continue from T (``tree_position_ids``); every trunk / branch is its own THD segment, so the
per-segment zigzag CP split needs no padding inside a trunk.  Loss weights live on branch tokens only, so the step
objective is unchanged.

Attention semantics of the tree (``tree_attention_mask``, a dense reference): a branch token sees its group's whole
trunk plus the earlier tokens of its own branch; a trunk token sees the earlier trunk tokens.  The Megatron backend
(``model.py``) does not build this mask: sequence-mixing modules run on a *virtual* ordinary pack derived from the
tree, so every existing attention kernel runs unchanged on documents that are exactly the original samples:

  'full': one document per sample = trunk + branch_i (real tokens contiguous from the document start, so positions /
          RoPE / block grids / conv windows are identical to the flat path);
  'halo': the trunk document once + one document per sample = last h trunk tokens + branch_i (h >= the receptive field
          of a short causal conv; used by the optional GDN state-fork mode).

expand (tree -> virtual) is an index_select (backward: index_add, i.e. the n sibling copies of a trunk token sum their
gradients); contract (virtual -> tree) picks one canonical copy per tree token (trunk rows from sample 0's document /
the trunk document, branch rows from their own document).

Nothing here imports torch except ``pack_tree``; ``model.py`` turns the maps into per-rank device index tensors.
"""
import numpy as np


def _round_up(x, m):
    return -(-x // m) * m


class TreeLayout:
    """Global (pre-CP-split) tree layout of one micro-batch and its virtual layouts."""

    def __init__(self, groups, cp, tp, pad_multiple=128, halo=32, align=None):
        """groups: list of (T, [branch_padded_len, ...]) in pack order (T and every branch length a multiple of
        ``align``, default 2*cp; a group with T == 0 is a plain sample).  A trailing pad segment makes the CP-local
        length a multiple of ``tp * pad_multiple`` (``pad_multiple=1``: only the TP multiple)."""
        align = align or 2 * cp
        self.cp, self.tp, self.pad_multiple, self.align = cp, tp, pad_multiple, align
        assert halo % align == 0, (halo, align)
        self.halo = halo
        seg = []          # (start, length, kind, group, member)
        pos = 0
        self.groups = []
        for g, (T, blens) in enumerate(groups):
            assert T % align == 0 and all(b % align == 0 and b > 0 for b in blens), (T, blens)
            ginfo = {'T': T, 'trunk': None, 'branches': []}
            if T:
                seg.append((pos, T, 'trunk', g, -1))
                ginfo['trunk'] = pos
                pos += T
            for m, b in enumerate(blens):
                seg.append((pos, b, 'branch', g, m))
                ginfo['branches'].append((pos, b))
                pos += b
            self.groups.append(ginfo)
        self.real_total = pos
        pad = self._trailing_pad(pos)
        if pad:
            seg.append((pos, pad, 'pad', -1, -1))
            pos += pad
        self.total = pos
        self.segments = seg
        self.cu = [s[0] for s in seg] + [pos]
        self.num_samples = sum(1 for s in seg if s[2] != 'pad')
        self.trailing = (self.real_total, pad)
        self.virtual = {k: self._build_virtual(k) for k in ('full', 'halo')}
        # max document length over every view (rotary tables are sized from it)
        self.max_seqlen = max(max(b - a for a, b in zip(v['cu'][:-1], v['cu'][1:])) for v in self.virtual.values())

    def _trailing_pad(self, total):
        local = total // self.cp
        pad_local = (-local) % (self.tp * self.pad_multiple)
        return pad_local * self.cp

    def _build_virtual(self, kind):
        docs = []          # (src tree positions, canonical mask) per virtual document
        for g in self.groups:
            T, t0 = g['T'], g['trunk']
            trunk = np.arange(t0, t0 + T) if T else np.zeros(0, dtype=np.int64)
            if kind == 'halo' and T:
                docs.append((trunk, np.ones(T, dtype=bool)))
            for m, (b0, b) in enumerate(g['branches']):
                br = np.arange(b0, b0 + b)
                if kind == 'full':
                    src = np.concatenate([trunk, br])
                    canon = np.concatenate([np.full(T, m == 0), np.ones(b, dtype=bool)])
                else:
                    h = min(self.halo, T)
                    src = np.concatenate([trunk[T - h:], br])
                    canon = np.concatenate([np.zeros(h, dtype=bool), np.ones(b, dtype=bool)])
                docs.append((src, canon))
        cu = [0]
        for src, _ in docs:
            cu.append(cu[-1] + len(src))
        real = cu[-1]
        pad = self._trailing_pad(real)
        tree_pad_start, tree_pad_len = self.trailing
        v_src = np.concatenate([d[0] for d in docs] + [np.zeros(0, dtype=np.int64)]).astype(np.int64)
        canon = np.concatenate([d[1] for d in docs] + [np.zeros(0, dtype=bool)])
        if pad:
            # the virtual trailing pad reads the tree's trailing pad tokens (cycled), else any token: it is its own
            # document and nothing reads its output, so its value never reaches a real token
            src = (tree_pad_start + np.arange(pad) % tree_pad_len) if tree_pad_len else np.zeros(pad, dtype=np.int64)
            v_src = np.concatenate([v_src, src])
            canon = np.concatenate([canon, np.zeros(pad, dtype=bool)])
            cu.append(cu[-1] + pad)
        total = cu[-1]
        assert total % (2 * self.cp) == 0 and (total // self.cp) % self.tp == 0
        # pad rows that read a REAL row of the other pack (only when that pack has no pad of its own) are zeroed
        # after the index_select (model.RankPlan), so no gradient is index_added into a real row
        v_zero = np.zeros(total, dtype=bool)
        if pad and not tree_pad_len:
            v_zero[real:] = True
        # contract map: every tree token -> one (canonical) virtual position
        t_src = np.full(self.total, -1, dtype=np.int64)
        idx = np.nonzero(canon)[0]
        t_src[v_src[idx]] = idx
        if tree_pad_len:
            t_src[tree_pad_start:] = (real + np.arange(tree_pad_len) % pad) if pad else 0
        t_zero = np.zeros(self.total, dtype=bool)
        if tree_pad_len and not pad:
            t_zero[tree_pad_start:] = True
        assert (t_src >= 0).all(), 'every tree token needs a canonical virtual copy'
        return {'cu': cu, 'v_src': v_src, 't_src': t_src, 'canon': canon, 'total': total, 'num_samples': len(docs),
                'v_zero': v_zero, 't_zero': t_zero,
                'max_seqlen': max(b - a for a, b in zip(cu[:-1], cu[1:]))}


# ---- reference semantics (position ids, dense attention mask) -------------------------------------------------

def tree_position_ids(layout):
    """Position id of every tree row: trunk 0..T-1, branch T..T+b-1 (continuing from the trunk), pad 0..pad-1."""
    pos = np.zeros(layout.total, dtype=np.int64)
    for g in layout.groups:
        T = g['T']
        if T:
            pos[g['trunk']:g['trunk'] + T] = np.arange(T)
        for b0, b in g['branches']:
            pos[b0:b0 + b] = np.arange(T, T + b)
    a, pad = layout.trailing
    if pad:
        pos[a:a + pad] = np.arange(pad)
    return pos


def tree_attention_mask(layout):
    """Dense causal tree mask [total, total] (True = query row may attend key column).

    Branch row: its group's whole trunk + the earlier rows of its own branch; trunk / pad row: earlier rows of its own
    segment.  Reference for backends that take an explicit mask (SDPA / flex attention); the Megatron backend uses the
    equivalent virtual 'full' pack instead (tests check both against the flat path)."""
    n = layout.total
    m = np.zeros((n, n), dtype=bool)
    for start, length, kind, gi, _ in layout.segments:
        blk = np.tril(np.ones((length, length), dtype=bool))
        m[start:start + length, start:start + length] = blk
        if kind == 'branch':
            g = layout.groups[gi]
            if g['T']:
                m[start:start + length, g['trunk']:g['trunk'] + g['T']] = True
    return m


# ---- CP / SP index maps ------------------------------------------------------------------------------------------

def local_positions(cu, cp, r):
    """Full positions held by CP rank r under the per-segment zigzag split (each segment cut into 2*cp chunks, rank r
    holds chunks r and 2*cp-1-r)."""
    if cp == 1:
        return np.arange(cu[-1], dtype=np.int64)
    out = []
    for a, b in zip(cu[:-1], cu[1:]):
        n = b - a
        assert n % (2 * cp) == 0, (a, b, cp)
        c = n // (2 * cp)
        out.append(np.arange(a + r * c, a + (r + 1) * c))
        out.append(np.arange(a + (2 * cp - 1 - r) * c, a + (2 * cp - r) * c))
    return np.concatenate(out).astype(np.int64) if out else np.zeros(0, dtype=np.int64)


def gathered_inverse(cu, cp):
    """full position -> row in the rank-major CP all-gather of the local shards."""
    g = np.concatenate([local_positions(cu, cp, r) for r in range(cp)])
    inv = np.empty_like(g)
    inv[g] = np.arange(len(g))
    return inv


def rank_maps(layout, kind, cp_rank, tp_rank, sp):
    """Index maps for one rank.

    expand   : rows of the CP-gathered *tree* tensor (rank-major order of CP-local tensors, each the TP-gathered
               CP-local tensor) that form this rank's virtual local tensor (its SP chunk if sp).
    contract : rows of the CP-gathered *virtual* tensor that form this rank's tree local tensor (SP chunk if sp).
    expand_cp / contract_cp: the same for CP-local tensors that are not SP-split (q/k/v, rotary, ids, positions).
    *_zero   : rows to zero after the index_select (pad rows reading a real row of the other pack), None = none.
    """
    cp, tp = layout.cp, layout.tp
    v = layout.virtual[kind]
    tinv = gathered_inverse(layout.cu, cp)
    vinv = gathered_inverse(v['cu'], cp)
    v_local = local_positions(v['cu'], cp, cp_rank)
    t_local = local_positions(layout.cu, cp, cp_rank)
    expand_cp = tinv[v['v_src'][v_local]]
    contract_cp = vinv[v['t_src'][t_local]]
    ez_cp = v['v_zero'][v_local]
    cz_cp = v['t_zero'][t_local]
    if sp and tp > 1:
        nv, nt = len(v_local) // tp, len(t_local) // tp
        expand = expand_cp[tp_rank * nv:(tp_rank + 1) * nv]
        contract = contract_cp[tp_rank * nt:(tp_rank + 1) * nt]
        ez, cz = ez_cp[tp_rank * nv:(tp_rank + 1) * nv], cz_cp[tp_rank * nt:(tp_rank + 1) * nt]
    else:
        expand, contract, ez, cz = expand_cp, contract_cp, ez_cp, cz_cp

    def nz(m):
        return m if m.any() else None

    return {'expand': expand, 'contract': contract, 'expand_cp': expand_cp, 'contract_cp': contract_cp,
            'expand_zero': nz(ez), 'contract_zero': nz(cz), 'expand_cp_zero': nz(ez_cp), 'contract_cp_zero': nz(cz_cp),
            'virtual_local_len': len(v_local), 'tree_local_len': len(t_local)}


# ---- sibling grouping --------------------------------------------------------------------------------------------

def lcp(a, b):
    """Length of the longest common prefix of two 1-D integer sequences (numpy arrays or CPU tensors)."""
    a, b = np.asarray(a), np.asarray(b)
    n = min(len(a), len(b))
    if n == 0:
        return 0
    neq = np.flatnonzero(a[:n] != b[:n])
    return int(neq[0]) if len(neq) else n


def trunk_align(cp):
    """THD alignment of a trunk under the per-segment zigzag CP split."""
    return 2 * cp


def trunk_len(prompt_len, cp):
    """Shared trunk length: largest multiple of trunk_align(cp) that leaves the last prompt token in every branch."""
    a = trunk_align(cp)
    return max(0, (prompt_len - 1) // a * a)


def group_by_prompt(seqs, plens, min_trunk=16):
    """slime rule.  seqs: 1-D int arrays (one micro-batch, in order); plens: prompt length per sample.

    Groups = samples with identical prompt tokens (GRPO siblings), in order of first appearance; trunk T = longest
    common prefix of the group (it may extend into the response when siblings share their first response tokens),
    capped so every member keeps >= 1 branch token.  Groups with T < min_trunk become singletons.
    Returns [(T, [sample idx])] (T == 0: a plain sample)."""
    by_key, order = {}, []
    for i, (s, p) in enumerate(zip(seqs, plens)):
        p = int(p)
        k = ('solo', i) if p <= 0 else (p, np.asarray(s[:p]).tobytes())
        if k not in by_key:
            by_key[k] = []
            order.append(k)
        by_key[k].append(i)
    out = []
    for k in order:
        idx = by_key[k]
        if len(idx) < 2:
            out.append((0, idx))
            continue
        ref = seqs[idx[0]]
        common = len(ref)
        for i in idx[1:]:
            common = min(common, lcp(ref, seqs[i]))
        T = min(common, min(len(seqs[i]) for i in idx) - 1)
        if T < min_trunk:
            out.extend((0, [i]) for i in idx)
        else:
            out.append((T, idx))
    return out


def group_consecutive(seqs, align, min_trunk=None, min_frac=0.25):
    """verl rule.  Groups = maximal runs of consecutive samples whose common prefix gives an aligned trunk
    T >= max(min_trunk, min_frac * shorter length) (siblings, not just a shared system prompt); T = (lcp-1) aligned
    down.  Returns [(T, [sample idx])] in pack order."""
    min_trunk = min_trunk or 4 * align
    out = []
    i, n = 0, len(seqs)
    while i < n:
        j = i + 1
        common = len(seqs[i])
        while j < n:
            l2 = min(common, lcp(seqs[i], seqs[j]))
            if (l2 - 1) // align * align < max(min_trunk, min_frac * min(len(seqs[i]), len(seqs[j]))):
                break
            common = l2
            j += 1
        if j - i >= 2:
            out.append(((common - 1) // align * align, list(range(i, j))))
        else:
            out.append((0, [i]))
        i = j
    return out


def sibling_units(seqs, align, budget, min_frac=0.25, key_len=32):
    """Group samples sharing a prefix (any order) and cut each group into units that fit the token budget: a unit =
    trunk + as many branches as fit (a group too big for one micro-batch is split and its trunk repeated).
    Returns [(unit tree tokens, [sample idx])]."""
    buckets = {}
    for i, s in enumerate(seqs):
        buckets.setdefault(np.asarray(s[:key_len]).tobytes(), []).append(i)
    clusters = []
    for idx in buckets.values():
        reps = []
        for i in idx:
            best, bl = None, -1
            for c in reps:
                x = lcp(seqs[c[0]], seqs[i])
                if x > bl:
                    best, bl = c, x
            if best is not None and (bl - 1) // align * align >= max(4 * align, min_frac * min(len(seqs[best[0]]), len(seqs[i]))):
                best.append(i)
            else:
                reps.append([i])
        clusters += reps
    units = []
    for idx in clusters:
        if len(idx) < 2:
            units.append((len(seqs[idx[0]]), idx))
            continue
        common = len(seqs[idx[0]])
        for i in idx[1:]:
            common = min(common, lcp(seqs[idx[0]], seqs[i]))
        T = (common - 1) // align * align
        if T < 4 * align:
            units += [(len(seqs[i]), [i]) for i in idx]
            continue
        cur, size = [], 0
        for i in idx:
            add = len(seqs[i]) - T
            if cur and size + add > budget:
                units.append((size, cur))
                cur, size = [], 0
            if not cur:
                size = T
            cur.append(i)
            size += add
        units.append((size, cur))
    return units


# ---- sample dicts -> tree pack ---------------------------------------------------------------------------------

def pack_tree(group_samples, prompt_lens, cp, tp, pad_multiple=128, halo=32, zero_keys=None):
    """group_samples: list (one per group) of lists of sample dicts (1-D tensors over the sample's tokens, all the same
    length within a sample; 'input_ids' first P entries equal within a group).  prompt_lens: P per group.
    Returns (packed dict of [1, total] tensors incl. position_ids, meta, TreeLayout)."""
    import torch
    align = 2 * cp
    zero_keys = set(zero_keys or ('input_ids', 'labels', 'mask', 'weight', 'advantages'))
    keys = list(group_samples[0][0].keys())
    fields = {k: [] for k in keys}
    fields['position_ids'] = []
    spec = []
    flat_tokens = 0
    for members, P in zip(group_samples, prompt_lens):
        T = trunk_len(P, cp)
        ref = members[0]
        for s in members[1:]:
            assert torch.equal(s['input_ids'][:P], ref['input_ids'][:P]), 'group members must share the prompt'
        blens = []
        if T:
            for k in keys:
                v = ref[k][:T]
                if k in ('mask', 'weight'):
                    v = torch.zeros_like(v)
                fields[k].append(v)
            fields['position_ids'].append(torch.arange(T))
        for s in members:
            n = s['input_ids'].numel()
            flat_tokens += _round_up(n, align)
            b = n - T
            bp = _round_up(b, align)
            for k in keys:
                v = s[k][T:]
                fill = 0 if k in zero_keys else v[-1]
                fields[k].append(torch.cat((v, v.new_full((bp - b,), fill))))
            fields['position_ids'].append(torch.arange(T, T + bp))
            blens.append(bp)
        spec.append((T, blens))
    layout = TreeLayout(spec, cp, tp, pad_multiple, halo)
    pad = layout.trailing[1]
    if pad:
        for k in keys:
            fields[k].append(fields[k][-1].new_zeros(pad))
        fields['position_ids'].append(torch.arange(pad))
    packed = {k: torch.cat(v)[None, :] for k, v in fields.items()}
    assert packed['input_ids'].shape[1] == layout.total
    meta = {'cu_seqlens': layout.cu, 'num_samples': layout.num_samples, 'max_seqlen': layout.max_seqlen,
            'pad_tokens_global': pad, 'total_tokens_global': layout.total,
            'tree_tokens_global': layout.total, 'flat_tokens_global': flat_tokens,
            'samples': sum(len(m) for m in group_samples), 'groups': len(group_samples)}
    return packed, meta, layout


def self_test(seed=0, trials=20):
    """Index-map invariants on random layouts (no torch).  Returns {'passed': True} or raises AssertionError."""
    rng = np.random.default_rng(seed)
    for cp, tp in ((1, 1), (2, 2), (1, 2), (2, 1), (4, 2), (8, 1), (8, 2)):
        align = 2 * cp
        for _ in range(trials):
            groups = []
            for _g in range(rng.integers(1, 4)):
                P = int(rng.integers(1, 200))
                T = trunk_len(P, cp)
                blens = [_round_up(P - T + int(rng.integers(1, 90)), align) for _ in range(int(rng.integers(1, 5)))]
                groups.append((T, blens))
            lay = TreeLayout(groups, cp, tp, pad_multiple=4, halo=2 * align)
            x = rng.standard_normal(lay.total)            # a tree "activation"
            for kind in ('full', 'halo'):
                v = lay.virtual[kind]
                vx = x[v['v_src']]
                R = lay.real_total
                assert np.array_equal(vx[v['t_src']][:R], x[:R])          # contract(expand(x)) == x on real rows
                for r in range(cp):
                    tl = local_positions(lay.cu, cp, r)
                    gathered_tree = np.concatenate([x[local_positions(lay.cu, cp, q)] for q in range(cp)])
                    vl = local_positions(v['cu'], cp, r)
                    gathered_v = np.concatenate([vx[local_positions(v['cu'], cp, q)] for q in range(cp)])
                    for t in range(tp):
                        m = rank_maps(lay, kind, r, t, sp=True)
                        nv = len(vl) // tp
                        assert np.array_equal(gathered_tree[m['expand']], vx[vl[t * nv:(t + 1) * nv]])
                        nt = len(tl) // tp
                        assert np.array_equal(gathered_v[m['contract']], vx[v['t_src']][tl[t * nt:(t + 1) * nt]])
            # 'full' documents are the original samples: trunk + branch_i contiguous from the document start
            v = lay.virtual['full']
            d = 0
            for g in lay.groups:
                for b0, b in g['branches']:
                    a = v['cu'][d]
                    trunk = np.arange(g['trunk'], g['trunk'] + g['T']) if g['T'] else np.zeros(0, np.int64)
                    assert np.array_equal(v['v_src'][a:a + g['T'] + b], np.concatenate([trunk, np.arange(b0, b0 + b)]))
                    d += 1
    return {'passed': True}
