"""GPU numerics test: slime / verl adapters vs the unmodified framework path on a small Megatron GPT.

torchrun --nproc-per-node=<tp*cp> test_adapters_gpu.py --fw slime|verl [--tp 2 --sp] [--cp 2] [--allgather-cp]
         [--mode response|loss] [--temperature 0.8]
Runs the same micro-batch twice through the framework's own functions (slime: model.get_batch + model forward +
loss.get_log_probs_and_entropy; verl: get_mcore_engine_forward_fn(...) with MegatronEngineWithLMHead's
_lm_head_logits_processor), first stock, then after adapter install(), and compares per-sample response
log-probs / entropy, the output-layer weight grad and the grad of the decoder output (hidden states).
"""
import argparse
import os
from functools import partial
from types import SimpleNamespace

import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument('--fw', choices=['slime', 'verl'], required=True)
p.add_argument('--tp', type=int, default=1)
p.add_argument('--cp', type=int, default=1)
p.add_argument('--sp', action='store_true')
p.add_argument('--allgather-cp', action='store_true')
p.add_argument('--mode', default='response', choices=['response', 'loss'])
p.add_argument('--temperature', type=float, default=1.0)
p.add_argument('--vocab', type=int, default=8192)
p.add_argument('--hidden', type=int, default=256)
p.add_argument('--layers', type=int, default=2)
p.add_argument('--nseq', type=int, default=6)
p.add_argument('--no-attn', action='store_true', help='replace the decoder by a pointwise op (layouts TE CP cannot run)')
p.add_argument('--with-entropy-grad', action='store_true')
p.add_argument('--zero-mask', action='store_true', help='loss_mask all 0 (with --mode loss: no row selected anywhere)')
p.add_argument('--attn-backend', default='auto', help='megatron AttnBackend name (auto/fused/flash/unfused/local)')
p.add_argument('--chunk', default=None, help='RESPONSE_LOGITS_CHUNK for the adapter run (e.g. 64 = chunked op, '
               '64-row blocks; off = compact logits)')
p.add_argument('--chunk-ab', action='store_true', help='baseline = adapter with RESPONSE_LOGITS_CHUNK=off (compact '
               'logits) instead of the stock framework path; candidate = adapter with --chunk')
args = p.parse_args()
os.environ['RESPONSE_LOGITS_ROWS'] = args.mode
if args.chunk is not None:
    os.environ['RESPONSE_LOGITS_CHUNK'] = args.chunk

dist.init_process_group('nccl')
rank = dist.get_rank()
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
ps.initialize_model_parallel(tensor_model_parallel_size=args.tp, context_parallel_size=args.cp)
model_parallel_cuda_manual_seed(1234)
torch.manual_seed(1234)

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.enums import AttnBackend

cfg = TransformerConfig(num_layers=args.layers, hidden_size=args.hidden, num_attention_heads=4, ffn_hidden_size=4 * args.hidden,
                        bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
                        tensor_model_parallel_size=args.tp, context_parallel_size=args.cp,
                        sequence_parallel=args.sp and args.tp > 1, hidden_dropout=0.0, attention_dropout=0.0,
                        calculate_per_token_loss=False, attention_backend=AttnBackend[args.attn_backend])
gpt = GPTModel(cfg, get_gpt_layer_with_transformer_engine_spec(), vocab_size=args.vocab, max_sequence_length=16384,
               position_embedding_type='rope', share_embeddings_and_output_weights=False).cuda()
model = Float16Module(cfg, gpt)
if args.no_attn:
    gpt.decoder.forward = lambda hidden_states=None, *a, **k: torch.tanh(hidden_states) * 2.0
HID = {}


def _keep_hidden(module, inputs, output):  # must return None (a return value would replace the output)
    if output.requires_grad:
        output.retain_grad()
    HID['h'] = output


gpt.decoder.register_forward_hook(_keep_hidden)

# ---- synthetic rollout (identical on all ranks) ----
g = torch.Generator().manual_seed(7)
samples = []
for i in range(args.nseq):
    total = int(torch.randint(200, 1400, (1,), generator=g))
    resp = max(4, int(total * float(torch.empty(1).uniform_(0.05, 0.3, generator=g))))
    toks = torch.randint(1, args.vocab, (total,), generator=g)
    lm = torch.ones(resp, dtype=torch.int32)
    for _ in range(2):  # tool-output spans inside the response: loss_mask 0
        a = int(torch.randint(0, resp, (1,), generator=g)); b = min(resp, a + int(torch.randint(1, max(2, resp // 4), (1,), generator=g)))
        lm[a:b] = 0
    if args.zero_mask:
        lm.zero_()
    samples.append((toks, total, resp, lm))


def wvec(i, n, salt):
    gg = torch.Generator().manual_seed(1000 * i + salt)
    return torch.randn(n, generator=gg)


def _with_chunk(value, fn):
    if value is None:
        return fn()
    old = os.environ.get('RESPONSE_LOGITS_CHUNK')
    os.environ['RESPONSE_LOGITS_CHUNK'] = value
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop('RESPONSE_LOGITS_CHUNK', None)
        else:
            os.environ['RESPONSE_LOGITS_CHUNK'] = old


def run_slime():
    import slime.backends.megatron_utils.loss as L
    import slime.backends.megatron_utils.model as M
    from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp
    from slime.backends.megatron_utils.data import DataIterator
    sargs = SimpleNamespace(allgather_cp=args.allgather_cp, rollout_temperature=args.temperature, log_probs_chunk_size=-1,
                            entropy_coef=0.01 if args.with_entropy_grad else 0.0, loss_type='policy_loss')
    rollout = {'tokens': [s[0].cuda() for s in samples], 'total_lengths': [s[1] for s in samples],
               'response_lengths': [s[2] for s in samples], 'loss_masks': [s[3].cuda() for s in samples]}

    def once(train=True):
        it = DataIterator(rollout, [list(range(len(samples)))])
        batch = M.get_batch(it, ['tokens', 'loss_masks', 'total_lengths', 'response_lengths'], 128, args.allgather_cp)
        model.zero_grad(set_to_none=True)
        model.train(train)
        with torch.set_grad_enabled(train):
            logits = model(input_ids=batch['tokens'], position_ids=None, attention_mask=None, labels=None,
                           packed_seq_params=batch['packed_seq_params'], loss_mask=batch['full_loss_masks'])
            _, res = M.__dict__.get('_rl_dummy', L.get_log_probs_and_entropy)(
                logits, args=sargs, unconcat_tokens=batch['unconcat_tokens'], total_lengths=batch['total_lengths'],
                response_lengths=batch['response_lengths'], with_entropy=True)
        lps, ents, masks = res['log_probs'], res['entropy'], []
        for i, (lp, s) in enumerate(zip(lps, samples)):
            m = s[3].cuda().float()
            if args.cp > 1:
                m = slice_log_prob_with_cp(m, s[1], s[2])
            assert m.numel() == lp.numel(), (m.numel(), lp.numel())
            masks.append(m)
        out = {'lp': [x.detach().float() for x in lps], 'ent': [x.detach().float() for x in ents], 'mask': masks,
               'logits_shape': tuple(logits.shape)}
        if train:
            loss = sum(((lp * wvec(i, lp.numel(), 1).cuda() + en * wvec(i, en.numel(), 2).cuda()) * m).sum()
                       for i, (lp, en, m) in enumerate(zip(lps, ents, masks)))
            loss.backward()
            out['gw'] = gpt.output_layer.weight.grad.detach().float().clone()
            out['gh'] = HID['h'].grad.detach().float().clone()
            out['ge'] = gpt.embedding.word_embeddings.weight.grad.detach().float().clone()
        return out

    from ai_search_plugin.response_logits.adapters import slime as rl
    if args.chunk_ab:
        rl.install(sargs)
    base, base_eval = _with_chunk('off' if args.chunk_ab else None, lambda: (once(True), once(False)))
    st = rl.install(sargs)
    if rank == 0:
        print('install:', st)
    new, new_eval = once(True), once(False)
    return base, base_eval, new, new_eval, rl.stats()


def run_verl():
    import verl.models.mcore as vm
    import verl.utils.megatron_utils as vmu
    if not all(isinstance(c, type) for c in vmu.ALL_MODULE_WRAPPER_CLASSNAMES):
        # env shim: verl (2026-10) pins mcore 0.19 where MegatronFSDP is a class; in 0.20 it is not
        from megatron.core.distributed import DistributedDataParallel as DDP
        vmu.unwrap_model.__defaults__ = ((DDP, Float16Module),)
    from verl.utils import torch_functional as verl_F
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
    fake = SimpleNamespace(engine_config=SimpleNamespace(entropy_from_logits_with_chunking=False,
                                                         entropy_from_logits_chunk_size=2048))
    hf = SimpleNamespace(architectures=['Qwen3ForCausalLM'])
    ids = torch.nested.nested_tensor([s[0] for s in samples], layout=torch.jagged).cuda()
    R = max(s[2] for s in samples)
    loss_mask = torch.zeros(len(samples), R, dtype=torch.bool)
    ram = torch.zeros(len(samples), R, dtype=torch.int64)
    for i, s in enumerate(samples):
        loss_mask[i, :s[2]] = s[3].bool(); ram[i, :s[2]] = 1
    loss_mask, ram = loss_mask.cuda(), ram.cuda()

    def once(train=True):
        proc = partial(MegatronEngineWithLMHead._lm_head_logits_processor, fake, calculate_sum_pi_squared=False,
                       calculate_entropy=True, distillation_use_topk=False, distillation_only=False,
                       logits_processor_func=None, batch=None, data_format='thd', cp_layout='zigzag', local_cp_size=None)
        temp = verl_F.expand_as_nested(torch.full((len(samples),), args.temperature, device='cuda'), ids)
        lpa = {'label': ids.clone(), 'temperature': temp, 'loss_mask': loss_mask, 'response_attention_mask': ram}
        model.zero_grad(set_to_none=True)
        model.train(train)
        with torch.set_grad_enabled(train):
            out = vm.get_mcore_engine_forward_fn(hf)(model, ids, {}, logits_processor=proc, logits_processor_args=lpa,
                                                     data_format='thd', cp_layout='zigzag')
        lpv, env = out['log_probs'].values(), out['entropy'].values()
        lps, ents, masks = [], [], []
        off = 0
        for s in samples:  # response rows of the unpacked full-length output (= no_padding_2_padding)
            end = off + s[1]
            lps.append(lpv[end - s[2] - 1:end - 1]); ents.append(env[end - s[2] - 1:end - 1])
            masks.append(s[3].cuda().float())
            off = end
        res = {'lp': [x.detach().float() for x in lps], 'ent': [x.detach().float() for x in ents], 'mask': masks}
        if train:
            loss = sum(((lp * wvec(i, lp.numel(), 1).cuda() + en * wvec(i, en.numel(), 2).cuda()) * m).sum()
                       for i, (lp, en, m) in enumerate(zip(lps, ents, masks)))
            loss.backward()
            res['gw'] = gpt.output_layer.weight.grad.detach().float().clone()
            res['gh'] = HID['h'].grad.detach().float().clone()
            res['ge'] = gpt.embedding.word_embeddings.weight.grad.detach().float().clone()
        return res

    from ai_search_plugin.response_logits.adapters import verl as rl
    if args.chunk_ab:
        rl.install()
    base, base_eval = _with_chunk('off' if args.chunk_ab else None, lambda: (once(True), once(False)))
    st = rl.install()
    if rank == 0:
        print('install:', st)
    new, new_eval = once(True), once(False)
    return base, base_eval, new, new_eval, rl.stats()


base, base_eval, new, new_eval, st = (run_slime if args.fw == 'slime' else run_verl)()
from ai_search_plugin import response_logits as core


def mx(t):
    t = torch.tensor([float(t)], device='cuda')
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def cmp_lists(a, b, masks, key):
    d = 0.0
    for x, y, m in zip(a[key], b[key], masks):
        assert x.shape == y.shape, (key, x.shape, y.shape)
        sel = m > 0 if args.mode == 'loss' else torch.ones_like(m, dtype=torch.bool)
        if sel.any():
            d = max(d, (x[sel] - y[sel]).abs().max().item())
        if args.mode == 'loss' and (~sel).any():
            assert y[~sel].abs().max().item() == 0.0, 'masked rows must be 0 in loss mode'
    return mx(d)


def rel(a, b):
    if b.norm() == 0:
        return mx((a - b).norm().item())
    return mx(((a - b).norm() / b.norm()).item())


masks = base['mask']
res = {
    'lp_maxabs': cmp_lists(base, new, masks, 'lp'), 'ent_maxabs': cmp_lists(base, new, masks, 'ent'),
    'lp_eval_maxabs': cmp_lists(base_eval, new_eval, masks, 'lp'), 'ent_eval_maxabs': cmp_lists(base_eval, new_eval, masks, 'ent'),
    'grad_out_w_rel': rel(new['gw'], base['gw']), 'grad_out_w_maxabs': mx((new['gw'] - base['gw']).abs().max()),
    'grad_hidden_rel': rel(new['gh'], base['gh']), 'grad_hidden_maxabs': mx((new['gh'] - base['gh']).abs().max()),
    'grad_embed_rel': rel(new['ge'], base['ge']),
    'lp_absmax': mx(max(x.abs().max().item() for x in base['lp'] if x.numel())),
}
core_stats = core.stats()
if rank == 0:
    tag = f"fw={args.fw} tp={args.tp} sp={args.sp} cp={args.cp} allgather_cp={args.allgather_cp} mode={args.mode} T={args.temperature} no_attn={args.no_attn}"
    print('RESULT', tag)
    for k, v in res.items():
        print(f'  {k:20s} {v:.3e}')
    print('  adapter stats', st, 'core rows', {k: v for k, v in core_stats.items() if k.startswith('row')})
    ok = (res['lp_maxabs'] < 5e-2 and res['ent_maxabs'] < 5e-2 and res['grad_out_w_rel'] < 2e-2
          and res['grad_hidden_rel'] < 2e-2 and core_stats['row_calls'] > 0)
    from ai_search_plugin.response_logits import chunked as _ch
    print('  chunked', _ch.stats(), 'row_chunked', core_stats.get('row_chunked'), 'ineligible',
          core_stats.get('chunk_ineligible'))
    if args.chunk_ab:  # compact logits vs chunked op: same math, only fp32 reduction order / GEMM row blocks differ
        # (bf16 grads flip by one ulp where the GEMM accumulation order changes: see test_chunked_op noise floor)
        ok = (res['lp_maxabs'] < 1e-5 and res['ent_maxabs'] < 1e-5 and res['lp_eval_maxabs'] < 1e-5
              and res['grad_out_w_rel'] < 2e-3 and res['grad_hidden_rel'] < 2e-3 and core_stats['row_chunked'] > 0)
    if args.zero_mask:  # dummy-row path: must run, grads must stay exactly zero
        ok = ok and res['grad_out_w_maxabs'] == 0.0 and res['grad_hidden_maxabs'] == 0.0
    print('PASS' if ok else 'FAIL', tag)
dist.barrier()
dist.destroy_process_group()
