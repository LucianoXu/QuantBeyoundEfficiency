from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F


def _suffix_embeds(suffix_ids: torch.Tensor, E: torch.Tensor, l_suf: int, V: int):
    # embed the suffix
    one_hot = torch.zeros(l_suf, V, device=E.device, dtype=E.dtype)
    one_hot.scatter_(1, suffix_ids.unsqueeze(1), 1.0)
    one_hot.requires_grad_(True)
    return one_hot, one_hot @ E


def _target_ce_from_embeds(p: dict[str, Any], suffix_embeds: torch.Tensor, model) -> torch.Tensor:
    # teacher-forced target
    embeds = torch.cat(
        [p["before_embeds"], suffix_embeds, p["after_embeds"], p["target_embeds"]], dim=0
    ).unsqueeze(0)
    nt = p["n_target"]
    logits = model(inputs_embeds=embeds, logits_to_keep=nt + 1).logits[0]
    return F.cross_entropy(logits[:nt].float(), p["target_ids"])


@torch.no_grad()
def _target_ce_batch(p: dict[str, Any], cand: torch.Tensor, model, V: int, chunk: int) -> torch.Tensor:
    # cross-entropy loss on the batch of candidates
    B = cand.shape[0]
    before = p["before_ids"].unsqueeze(0).expand(B, -1)
    after = p["after_ids"].unsqueeze(0).expand(B, -1)
    tgt = p["target_ids"].unsqueeze(0).expand(B, -1)
    nt = p["n_target"]
    ces = []

    for i in range(0, B, chunk):
        ids = torch.cat([before[i:i + chunk], cand[i:i + chunk], after[i:i + chunk], tgt[i:i + chunk]], dim=1)
        b = ids.shape[0]
        pred = model(input_ids=ids, logits_to_keep=nt + 1).logits[:, :nt, :]
        ce = F.cross_entropy(
            pred.reshape(-1, V).float(), 
            p["target_ids"].repeat(b),
            reduction="none"
        ).view(b, nt).mean(dim=1)
        ces.append(ce)
    return torch.cat(ces)


def build_objective(kind: str, model_full, l_suf: int, atk: dict[str, Any], tokenizer,
                    model_q=None) -> AttackObjective:
    
    E = model_full.get_input_embeddings().weight
    chunk = atk.get("chunk", 256)

    if kind == "single":
        return SingleObjective(model_full, E, l_suf, chunk)
    
    if kind == "differential":
        if model_q is None:
            raise ValueError("differential objective requires model_q (the fake-quant proxy)")
        
        return DifferentialObjective(
            model_full, model_q, E, l_suf, chunk,
            lam_refuse=atk.get("lam_refuse", 1.0),
            a_margin=atk.get("a_margin", 0.5), b_margin=atk.get("b_margin", 2.0))
    
    raise ValueError(f"Unknown objective {kind!r} (single|differential)")


class AttackObjective(ABC):
    def __init__(self, E: torch.Tensor, l_suf: int, chunk: int = 256):
        self.E = E
        self.V, self.d = E.shape
        self.l_suf = l_suf
        self.chunk = chunk

    @abstractmethod
    def token_gradient(self, suffix_ids: torch.Tensor, active: list[dict[str, Any]]) -> torch.Tensor: ...

    @abstractmethod
    def batch_loss(self, cand: torch.Tensor, active: list[dict[str, Any]]) -> torch.Tensor: ...

    def diagnostics(self, suffix_ids: torch.Tensor, active: list[dict[str, Any]]) -> dict[str, float]:
        # logline for one suffix
        return {"loss": float(self.batch_loss(suffix_ids.unsqueeze(0), active)[0])}


class SingleObjective(AttackObjective):
    def __init__(self, model, E, l_suf, chunk=256):
        super().__init__(E, l_suf, chunk)
        self.model = model

    def token_gradient(self, suffix_ids, active):
        g_sum = None
        for p in active:
            one_hot, se = _suffix_embeds(suffix_ids, self.E, self.l_suf, self.V)
            loss = _target_ce_from_embeds(p, se, self.model)
            self.model.zero_grad(set_to_none=True)
            loss.backward()
            g = one_hot.grad.detach()
            g = g / g.norm().clamp_min(1e-8)
            g_sum = g if g_sum is None else g_sum + g
        return g_sum

    def batch_loss(self, cand, active):
        total = None
        for p in active:
            ce = _target_ce_batch(p, cand, self.model, self.V, self.chunk)
            total = ce if total is None else total + ce
        return total


class DifferentialObjective(AttackObjective):
    '''
    Maximizes the difference between fulfillment in model_full and model_q
    '''

    def __init__(self, model_full, model_q, E, l_suf, chunk=256,
                 lam_refuse: float = 1.0, a_margin: float = 0.5, b_margin: float = 2.0):
        super().__init__(E, l_suf, chunk)
        self.model_full = model_full
        self.model_q = model_q
        self.lam_refuse = lam_refuse
        self.a_margin, self.b_margin = a_margin, b_margin

    def batch_loss(self, cand, active):
        total = None
        for p in active:
            ce_q = _target_ce_batch(p, cand, self.model_q, self.V, self.chunk)
            ce_f = _target_ce_batch(p, cand, self.model_full, self.V, self.chunk)
            L = F.relu(ce_q - self.a_margin) + self.lam_refuse * F.relu(self.b_margin - ce_f)
            total = L if total is None else total + L
        return total


    def token_gradient(self, suffix_ids, active):
        # partial backward propagation
        one_hot = torch.zeros(self.l_suf, self.V, device=self.E.device, dtype=self.E.dtype)
        one_hot.scatter_(1, suffix_ids.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)
        
        for model, is_quant in ((self.model_q, True), (self.model_full, False)):
            for p in active:
                se = one_hot @ self.E
                nt = p["n_target"]
                embeds = torch.cat(
                    [p["before_embeds"], se, p["after_embeds"], p["target_embeds"]], dim=0).unsqueeze(0)
                logits = model(inputs_embeds=embeds, logits_to_keep=nt + 1).logits[0][:nt]
                ce = F.cross_entropy(logits.float(), p["target_ids"])
                partial = F.relu(ce - self.a_margin) if is_quant else self.lam_refuse * F.relu(self.b_margin - ce)
                partial.backward()
        return one_hot.grad.detach()

    @torch.no_grad()
    def diagnostics(self, suffix_ids, active):
        c = suffix_ids.unsqueeze(0)
        n = len(active)
        lq = sum(float(_target_ce_batch(p, c, self.model_q, self.V, self.chunk)[0]) for p in active)
        lf = sum(float(_target_ce_batch(p, c, self.model_full, self.V, self.chunk)[0]) for p in active)
        
        return {
            "loss": float(self.batch_loss(c, active)[0]),
            "L_tgt_quant": lq / n, 
            "L_tgt_full": lf / n, 
            "gap_full_minus_quant": (lf - lq) / n
        }
