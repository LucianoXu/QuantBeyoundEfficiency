# The real int4 kernel is not differentiable
# This is a differentiable proxy of int4 quantization for the attack
# Verification of equivalence is in test.

""" Differentiable fake-quantization for modeling the int4 precision. """

from typing import Any
import torch
import torch.nn as nn


def quanto_qint4_group_size(in_features: int, base: int = 128) -> int | None:
    """calculate the group size/bounds for dimensions (introduced by quantization) """
    if in_features > base:
        gs = base
        while in_features % gs != 0 and gs > 32:
            gs -= 32
        return gs if in_features % gs == 0 else None
    return None


@torch.no_grad()
def fake_quantize_qint4(W: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the int4 quantization result. Differentiable.
    '''
    out_f, in_f = W.shape
    gs = quanto_qint4_group_size(in_f)
    Wg = W.reshape(out_f, -1, gs) if gs is not None else W.unsqueeze(1)
    qmin, qmax = -(2 ** 3), 2 ** 3 - 1      # signed int4: [-8, 7]    
    rmin = Wg.amin(dim=-1, keepdim=True)
    rmax = Wg.amax(dim=-1, keepdim=True)
    scale = ((rmax - rmin) / (qmax - qmin)).clamp_min(1e-8)
    shift = -rmin
    q = torch.clamp(torch.round((Wg + shift) / scale) + qmin, qmin, qmax)
    deq = scale * (q - qmin) - shift
    return deq.reshape(out_f, in_f).to(W.dtype)


@torch.no_grad()
def apply_fake_quant_(model: nn.Module, ignore: tuple[str, ...] = ("lm_head",)):
    '''
    In place quantization.
    '''
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not any(ig in name for ig in ignore):
            W = mod.weight.data
            deq = fake_quantize_qint4(W)
            mod.weight.data = deq


def load_fake_quant_model(model_name: str, 
                          device_map, 
                          dtype=torch.bfloat16,
                          ignore: tuple[str, ...] = ("lm_head",)):
    '''
    load the given model in int4 quantization.
    '''
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device_map).eval()
    model.requires_grad_(False)
    apply_fake_quant_(model, ignore=ignore)
    return model
