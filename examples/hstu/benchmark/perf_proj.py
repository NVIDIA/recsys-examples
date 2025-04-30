from typing import Any, Dict, List, Tuple

from sympy import factor, sympify

B = 32
d = 128
h = 8

T = 15914
H = d * h

sms = 132
t = max(1, min(sms * 64, T // 4))  # tile_name
linear_weight_dtype = "bfloat16"
act_dtype = "bfloat16"
norm_weight_dtype = "float16"


def factor_expression(expr_str):
    expr = sympify(expr_str)

    factored_expr = factor(expr)
    return str(factored_expr).replace(" ", "")


input_output_: Dict[str, Dict[str, Dict[str, Tuple[Tuple, str]]]] = {
    "ln fwd": {
        "input": {
            "x": ((T, H), "(T, H)"),
            "bias": ((H,), "(H,)"),
            "weight": ((H,), "(H,)"),
        },
        "output": {
            "ln_x": ((T, H), "(T, H)"),
            "mean_x": ((T,), "(T,)"),
            "std_x": ((T,), "(T,)"),
        },
    },
    "uqkv (bias+silu)": {
        "input": {
            "ln_x": ((T, H), "(T, H)"),
            "bias": ((4 * H,), "(4 * H,)"),
            "weight": ((H, 4 * H), "(H, 4 * H)"),
        },
        "output": {
            "u": ((T, H), "(T, H)"),
            "q": ((T, H), "(T, H)"),
            "k": ((T, H), "(T, H)"),
            "v": ((T, H), "(T, H)"),
        },
    },
    "q.contiguous": {
        "input": {
            "q": ((T, H), "(T, H)"),
        },
        "output": {
            "q_contiguous": ((T, H), "(T, H)"),
        },
    },
    "k.contiguous": {
        "input": {
            "k": ((T, H), "(T, H)"),
        },
        "output": {
            "k_contiguous": ((T, H), "(T, H)"),
        },
    },
    "v.contiguous": {
        "input": {
            "v": ((T, H), "(T, H)"),
        },
        "output": {
            "v_contiguous": ((T, H), "(T, H)"),
        },
    },
    "attn_fwd": {
        "input": {
            "q_contiguous": ((T, H), "(T, H)"),
            "k_contiguous": ((T, H), "(T, H)"),
            "v_contiguous": ((T, H), "(T, H)"),
            "seq_offsets": ((T,), "(T,)"),
        },
        "output": {
            "attn_output": ((T, H), "(T, H)"),
        },
    },
    "_ln_mul_dropout_fwd": {
        "input": {
            "attn_output": ((T, H), "(T, H)"),
            "bias": ((H,), "(H,)"),
            "weight": ((H,), "(H,)"),
            "u": ((T, H), "(T, H)"),
        },
        "output": {
            "ln_mul_dropout_fwd": ((T, H), "(T, H)"),
            "mean": ((T,), "(T,)"),
            "std": ((T,), "(T,)"),
        },
    },
    "proj fwd (redisual)": {
        "input": {
            "ln_mul_dropout_fwd": ((T, H), "(T, H)"),
            "x": ((T, H), "(T, H)"),
            "weight": ((H, H), "(H, H)"),
        },
        "output": {
            "proj_fwd": ((T, H), "(T, H)"),
        },
    },
    "proj dw": {
        "input": {
            "proj_fwd": ((T, H), "(T, H)"),
            "grad_output": ((T, H), "(T, H)"),
        },
        "output": {
            "proj_dw": ((H, H), "(H, H)"),
        },
    },
    "proj dx": {
        "input": {
            "weight": ((H, H), "(H, H)"),
            "grad_output": ((T, H), "(T, H)"),
        },
        "output": {
            "proj_dx": ((T, H), "(T, H)"),
        },
    },
    "_ln_mul_dropout_bwd dx_du": {
        "input": {
            "proj_dx": ((T, H), "(T, H)"),
            "attn_output": ((T, H), "(T, H)"),
            "u": ((T, H), "(T, H)"),
            "weight": ((H,), "(H,)"),
            "bias": ((H,), "(H,)"),
            "mean": ((T,), "(T,)"),
            "std": ((T,), "(T,)"),
        },
        "output": {
            "ln_mul_dropout_bwd_dx": ((T, H), "(T, H)"),
            "ln_mul_dropout_bwd_du": ((T, H), "(T, H)"),
            "_dweight": ((t, H), "(t, H)"),
            "_dbias": ((t, H), "(t, H)"),
        },
    },
    "_ln_mul_dropout_bwd dw_db": {
        "input": {
            "_dweight": ((t, H), "(t, H)"),
            "_dbias": ((t, H), "(t, H)"),
        },
        "output": {
            "dweight": ((H,), "(H,)"),
            "dbias": ((H,), "(H,)"),
        },
    },
    # "empty1"
    # "empty2"
    "attn bwd 1": {
        "input": {},
        "output": {
            "dq": ((T, H), "(T, H)"),
            "dk": ((T, H), "(T, H)"),
            "dv": ((T, H), "(T, H)"),
        },
    },
    "attn bwd 2": {
        "input": {
            "dq": ((T, H), "(T, H)"),
        },
        "output": {
            "dq_cvt": ((T, H), "(T, H)"),
        },
    },
    "cat": {
        "input": {
            "ln_mul_dropout_bwd_du": ((T, H), "(T, H)"),
            "dv": ((T, H), "(T, H)"),
            "dq": ((T, H), "(T, H)"),
            "dk": ((T, H), "(T, H)"),
        },
        "output": {
            "cat": ((T, 4 * H), "(T, 4 * H)"),
        },
    },
    "silu_backward": {
        "input": {
            "cat": ((T, 4 * H), "(T, 4 * H)"),
        },
        "output": {
            "dsilu": ((T, 4 * H), "(T, 4 * H)"),
        },
    },
    "uvqk db": {
        "input": {
            "dsilu": ((T, 4 * H), "(T, 4 * H)"),
        },
        "output": {
            "uvqk_db": ((4 * H,), "(4 * H,)"),
        },
    },
    "uvqk dw": {
        "input": {
            "dsilu": ((T, 4 * H), "(T, 4 * H)"),
            "ln_x": ((T, H), "(T, H)"),
        },
        "output": {
            "uvqk_dw": ((H, 4 * H), "(H, 4 * H)"),
        },
    },
    "uvqk dx": {
        "input": {
            "dsilu": ((T, 4 * H), "(T, 4 * H)"),
            "weight": ((H, 4 * H), "(H, 4 * H)"),
        },
        "output": {
            "uvqk_dx": ((T, H), "(T, H)"),
        },
    },
    "ln_dx": {
        "input": {
            "uvqk_dx": ((T, H), "(T, H)"),
            "weight": ((H,), "(H,)"),
            "ln_x": ((T, H), "(T, H)"),
            "mean_x": ((T,), "(T,)"),
            "std_x": ((T,), "(T,)"),
        },
        "output": {
            "ln_dx": ((T, H), "(T, H)"),
            "_ln_weight": ((t, H), "(t, H)"),
            "_ln_bias": ((t, H), "(t, H)"),
        },
    },
    "ln_dwdb": {
        "input": {
            "_ln_weight": ((t, H), "(t, H)"),
            "_ln_bias": ((t, H), "(t, H)"),
        },
        "output": {
            "ln_dw": ((H,), "(H,)"),
            "ln_db": ((H,), "(H,)"),
        },
    },
    "uvqk db (grad_add)": {
        "input": {
            "ln_dx": ((T, H), "(T, H)"),
            "grad_output": ((T, H), "(T, H)"),
        },
        "output": {
            "grad_input": ((T, H), "(T, H)"),
        },
    },
}

import numpy as np

# idx = [[], [], []]
# numels = []
# shapes = []
df_dict: Dict[str, List[Any]] = {
    "kernel": [],
    "in_or_out": [],
    "tensor": [],
    "shape": [],
    "shape_str": [],
    "numel": [],
    "numel_str": [],
}
for kernel, info in input_output_.items():
    for in_or_out, tensors in info.items():
        for tensor, (shape, shape_str) in tensors.items():
            df_dict["kernel"].append(kernel)
            df_dict["in_or_out"].append(in_or_out)
            df_dict["tensor"].append(tensor)
            df_dict["shape"].append("[" + ",".join(str(x) for x in shape) + "]")
            df_dict["shape_str"].append(shape_str)
            df_dict["numel"].append(np.prod(shape))
            numel_str = "*".join(
                shape_str.replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace(" ", "")
                .replace("*", "")
            )
            df_dict["numel_str"].append(numel_str)

import pandas as pd

df = pd.DataFrame(df_dict)
df.set_index(["kernel", "in_or_out", "tensor"], inplace=True)

df_grouped = (
    df[["numel", "shape_str", "numel_str"]]
    .groupby(level="kernel")
    .agg(
        {
            "numel": "sum",
            "numel_str": lambda x: factor_expression("+".join(x)),
            "shape_str": lambda x: "+".join(x),
        }
    )
)
with pd.ExcelWriter("input_output.xlsx") as writer:
    df.to_excel(writer, sheet_name="raw_data", index=True)
    df_grouped.to_excel(writer, sheet_name="grouped_data", index=True)
