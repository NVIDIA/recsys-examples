# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial, wraps

import torch


def forward_nvtx_push_hook(module, input, msg=""):
    torch.cuda.nvtx.range_push(msg)


def forward_nvtx_pop_hook(module, input, output):
    torch.cuda.nvtx.range_pop()


def backward_nvtx_push_hook(module, grad_output, msg=""):
    torch.cuda.nvtx.range_push(msg)


def backward_nvtx_pop_hook(module, grad_input, grad_output):
    torch.cuda.nvtx.range_pop()


def register_nvtx_for_module(module: torch.nn.Module, msg=""):
    module.register_forward_pre_hook(
        partial(forward_nvtx_push_hook, msg=msg + " forward")
    )
    module.register_forward_hook(forward_nvtx_pop_hook)
    module.register_full_backward_pre_hook(
        partial(backward_nvtx_push_hook, msg=msg + " backward")
    )
    module.register_full_backward_hook(backward_nvtx_pop_hook)


# default key_or_attr_name is "data" for tensor
def hook_setter(x, y, key_or_attr_name=["data"]):
    if not isinstance(key_or_attr_name, tuple) and not isinstance(
        key_or_attr_name, list
    ):
        key_or_attr_name = [key_or_attr_name]

    leaf_key_or_attr_name = key_or_attr_name[-1]
    current = x

    access_path = []
    for key in key_or_attr_name[:-1]:
        access_path.append(str(key))
        try:
            current = current[key]
        # default path for tensor
        except (TypeError, KeyError, IndexError):
            current = getattr(current, key)
        except AttributeError as e:
            raise e(f"Error accessing member {access_path} of {x}")
        except Exception as e:
            raise e(f"Error accessing member {access_path} of {x}")
    try:
        current[leaf_key_or_attr_name] = y
    # default path for tensor
    except (TypeError, KeyError, IndexError):
        setattr(current, leaf_key_or_attr_name, y)
    except AttributeError as e:
        raise e(f"Error setting attribute {key_or_attr_name} of {x} to {y}")
    except Exception as e:
        raise e(f"Error setting attribute {key_or_attr_name} of {x} to {y}")


def hook_getter(x, key_or_attr_name=["data"]):
    if not isinstance(key_or_attr_name, tuple) and not isinstance(
        key_or_attr_name, list
    ):
        key_or_attr_name = [key_or_attr_name]
    current = x
    access_path = []
    for key in key_or_attr_name:
        access_path.append(str(key))
        try:
            current = current[key]
        # default path for tensor
        except (TypeError, KeyError, IndexError):
            current = getattr(current, key)
        except AttributeError as e:
            raise e(f"Error accessing member {access_path} of {x}")
        except Exception as e:
            raise e(f"Error accessing member {access_path} of {x}")
    return current


def register_setter_and_getter_for_nvtx(forward_func, key_or_attr_name=["data"]):
    forward_func.hook_tensor_getter = partial(
        hook_getter, key_or_attr_name=key_or_attr_name
    )
    forward_func.hook_tensor_setter = partial(
        hook_setter, key_or_attr_name=key_or_attr_name
    )


class _NvtxRangePush(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, msg="None"):
        ctx.msg = msg
        torch.cuda.nvtx.range_push(msg)
        return input

    @staticmethod
    def backward(ctx, grad_in):
        torch.cuda.nvtx.range_pop()
        return grad_in, None


class _NvtxRangePop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, msg="None"):
        ctx.msg = msg
        torch.cuda.nvtx.range_pop()
        return input

    @staticmethod
    def backward(ctx, grad_in):
        msg = ctx.msg
        torch.cuda.nvtx.range_push(msg)
        return grad_in, None


class _binary_identity_op(torch.autograd.Function):
    """
    input must be tensor
    """

    @staticmethod
    def forward(ctx, input_l, input_r):
        return input_l, input_r

    @staticmethod
    def backward(ctx, grad_l, grad_r):
        return grad_l, grad_r


## usage
NvtxRangePush = _NvtxRangePush.apply
NvtxRangePop = _NvtxRangePop.apply
binary_identity_op = _binary_identity_op.apply


def output_nvtx_hook(nvtx_tag, backward=True, hook_key_or_attr_name="data"):
    def decorator_forward_only(module):
        @wraps(module)
        def forward(*args, **kwags):
            torch.cuda.nvtx.range_push(nvtx_tag)
            output = module(*args, **kwags)
            torch.cuda.nvtx.range_pop()
            return output

        return forward

    def decorator_nvtx(module_or_func):
        @wraps(module_or_func)
        def wrapper(*args, **kwargs):
            _placeholder = torch.empty(1, device="cuda").requires_grad_()
            hook_r = NvtxRangePush(_placeholder, nvtx_tag + " forward")
            output = module_or_func(*args, **kwargs)

            getter = getattr(
                wrapper,
                "hook_tensor_getter",
                partial(hook_getter, key_or_attr_name=hook_key_or_attr_name),
            )
            setter = getattr(
                wrapper,
                "hook_tensor_setter",
                partial(hook_setter, key_or_attr_name=hook_key_or_attr_name),
            )
            try:
                if isinstance(output, torch.Tensor):
                    hook_l = output
                else:
                    hook_l = getter(output)
                hook_l, _ = binary_identity_op(hook_l, hook_r)
                hook_l = NvtxRangePop(hook_l, nvtx_tag + " backward")
                if isinstance(output, torch.Tensor):
                    output = hook_l
                else:
                    setter(output, hook_l)
            except Exception:
                # silently ignore
                NvtxRangePop(hook_r, nvtx_tag + " backward")
            return output

        return wrapper

    if not backward:
        decorator = decorator_forward_only
    else:
        decorator = decorator_nvtx
    return decorator
