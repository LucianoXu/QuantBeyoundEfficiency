# Fidelity tests for the fake-quant proxy (src/attacks/proxy.py) against the REAL optimum.quanto
# qint4 kernel -- the whole differential attack rests on the proxy matching the deployed int4 model.
#
#   test_weight_level_bit_exact : proxy dequant == quanto qint4 dequant, bit-exact (deterministic).
#   test_e2e_forward            : proxy MODEL's forward tracks the real int4 model at least as well
#                                 as the full model does (needs a GPU + the model; skipped w/o CUDA).
#
# Both need optimum.quanto's C++ kernel, which is C++20 -> on Raven run under `module load gcc/13`.
# Run:  python -m src.tests.test_proxy      (or)      python src/tests/test_proxy.py

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

import torch

from src.attacks.proxy import fake_quantize_qint4, quanto_qint4_group_size, apply_fake_quant_

MODEL = "Qwen/Qwen3-4B"


def test_weight_level_bit_exact():
    """Our fake_quantize_qint4 reproduces quanto's real qint4 dequantized weights exactly."""
    from optimum.quanto import qint4
    from optimum.quanto.tensor.optimizers import MaxOptimizer
    from optimum.quanto.tensor.weights import quantize_weight

    opt = MaxOptimizer()
    torch.manual_seed(0)
    shapes = [(2560, 2560), (2560, 9728), (256, 160), (128, 96), (64, 100)]  # incl. gs-reduction + per-row fallback
    for dtype in (torch.float32, torch.bfloat16):
        for out_f, in_f in shapes:
            W = (torch.randn(out_f, in_f) * 0.02).to(dtype)
            gs = quanto_qint4_group_size(in_f)
            scale, shift = opt(W, qint4, axis=0, group_size=gs, zeropoint=False)
            W_real = quantize_weight(W, qint4, axis=0, scale=scale, shift=shift, group_size=gs).dequantize()
            d = (fake_quantize_qint4(W).float() - W_real.float()).abs().max().item()
            assert d == 0.0, f"weight mismatch dtype={dtype} shape=({out_f},{in_f}) max|delta|={d}"
    print("weight-level bit-exact vs quanto qint4: PASS")


@torch.no_grad()
def _target_logits(model, p, suffix):
    dev = model.device
    ids = torch.cat([p["before_ids"].to(dev), suffix.to(dev),
                     p["after_ids"].to(dev), p["target_ids"].to(dev)]).unsqueeze(0)
    nt = p["n_target"]
    return model(input_ids=ids, logits_to_keep=nt + 1).logits[0, :nt].float().cpu()  # (nt, V)


def test_e2e_forward(n_prompts: int = 6, l_suf: int = 20):
    """Proxy-model forward tracks the real Quanto-int4 model >= the full model (argmax on target positions)."""
    if not torch.cuda.is_available():
        print("e2e forward test: SKIP (no CUDA)")
        return

    from transformers import AutoModelForCausalLM
    from src.models.factory import model_factory
    from src.attacks import scaffold as S

    tok, full = model_factory({"model_name": MODEL, "quant_type": "bf16", "device_map": "auto"}, seed=0)
    full.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    train, _ = S.load_sorry_bench_prompts(n_prompts, 0, sample_seed=0)
    prompts = S.build_prompts(train, tok, full.get_input_embeddings(), l_suf, full.device)
    suffix = S.init_suffix_ids(tok, l_suf, full.device)

    proxy = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
    apply_fake_quant_(proxy)                                   # weights -> qint4 dequant (== deployed int4)
    _, real = model_factory({"model_name": MODEL, "quant_type": "int4", "device_map": "auto"}, seed=0)
    real.eval()

    agree_pr = agree_fr = n = 0
    for p in prompts:
        lf, lp, lr = (_target_logits(m, p, suffix) for m in (full, proxy, real))
        agree_pr += int((lp.argmax(-1) == lr.argmax(-1)).sum())
        agree_fr += int((lf.argmax(-1) == lr.argmax(-1)).sum())
        n += p["n_target"]
    print(f"e2e forward: proxy-vs-int4 argmax {agree_pr}/{n}={agree_pr / n:.1%}  "
          f"full-vs-int4 {agree_fr}/{n}={agree_fr / n:.1%}")
    assert agree_pr >= agree_fr, "proxy should track real int4 at least as well as the full model"
    print("e2e forward (proxy tracks int4 >= full): PASS")


if __name__ == "__main__":
    test_weight_level_bit_exact()
    test_e2e_forward()
