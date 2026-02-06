import os
import sys
from functools import wraps

import torch

def need_dump():
    dump = os.environ.get('DUMP_PAGED_HSTU_TENSORS', default="0")
    return True if dump in {"1", 1, "True", True, "ON", "on", "On"} else False

def dump_paged_hstu_forward_naive(frame_locals, dump_status):
    if "jagged_attn_output" in frame_locals and "jagged_attn_output" not in dump_status:
        layer_idx = frame_locals.get("self").layer_idx
        # dump Q
        v_Q = frame_locals.get("query")
        torch.save(v_Q, f"/tmp/query_layer{layer_idx}.pt")
        # dump KV
        v_K = frame_locals.get("key")
        v_V = frame_locals.get("value")
        v_metadata = frame_locals.get("kv_cache_metadata")
        if v_metadata is not None:
            v_num_cand = int(frame_locals.get("jd").num_candidates[0])
            v_last_empty = 32 - int(v_metadata.kv_last_page_len[0])
            v_Kcand = v_K[-v_num_cand:, ...]
            v_Vcand = v_V[-v_num_cand:, ...]
            
            v_table = v_metadata.kv_cache_table[layer_idx]
            v_Khist = v_table[v_metadata.kv_indices, 0, ...]
            v_Khist = v_Khist.view(-1, v_Khist.size(2), v_Khist.size(3))
            v_Vhist = v_table[v_metadata.kv_indices, 1, ...]
            v_Vhist = v_Vhist.view(-1, v_Vhist.size(2), v_Vhist.size(3))
            v_Khist = v_Khist[:v_Khist.size(0)-v_last_empty]
            v_Vhist = v_Vhist[:v_Vhist.size(0)-v_last_empty]
            v_Kall = torch.cat([v_Khist, v_Kcand])
            v_Vall = torch.cat([v_Vhist, v_Vcand])
            torch.save(v_Kall, f"/tmp/key_layer{layer_idx}.pt")
            torch.save(v_Vall, f"/tmp/value_layer{layer_idx}.pt")
        else:
            torch.save(v_K, f"/tmp/key_layer{layer_idx}.pt")
            torch.save(v_V, f"/tmp/value_layer{layer_idx}.pt")
        # dump Attention Output
        v_A = frame_locals.get("jagged_attn_output")
        torch.save(v_A, f"/tmp/attn_layer{layer_idx}.pt")

        dump_status.add("jagged_attn_output")

    if "layer_output" in frame_locals and "layer_output" not in dump_status:
        layer_idx = frame_locals.get("self").layer_idx
        v_O = frame_locals.get("layer_output")
        torch.save(v_O, f"/tmp/out_layer{layer_idx}.pt")

        dump_status.add("layer_output")


def dump(func_name, dump_func):
    dump_status = set()
    def tracer(frame, event, arg):
        nonlocal dump_status
        if frame.f_code.co_name == func_name:
            if event == "call":
                frame.f_trace_lines = True
                frame.f_trace_opcodes = False
            elif event == "line":
                dump_func(frame.f_locals, dump_status)
            elif event == "return":
                dump_status.clear()
        return tracer

    def decorator(func):
        if not need_dump():
            return func
        @wraps(func)
        def wrapper(*args, **kwargs):
            sys.settrace(tracer)
            result = func(*args, **kwargs)
            sys.settrace(None)
            return result
        return wrapper
    return decorator
