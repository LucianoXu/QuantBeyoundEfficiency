# GCG; Zou et al. 2023, Algorithm 2: optimize a universal adversarial

from typing import Any, Callable, Optional

import torch


def masked_topk_per_position(grad: torch.Tensor, topk: int, disallowed_mask: torch.Tensor) -> torch.Tensor:
    # mask disallowed tokens
    g = grad.clone()
    g[:, disallowed_mask] = float("inf")
    return (-g).topk(topk, dim=1).indices


def sample_position_swaps(suffix_ids: torch.Tensor, top_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
    l_suf = suffix_ids.shape[0]
    topk = top_indices.shape[1]
    device = suffix_ids.device
    cand = suffix_ids.unsqueeze(0).repeat(batch_size, 1)
    pos = torch.randint(0, l_suf, (batch_size,), device=device)
    pick = torch.randint(0, topk, (batch_size,), device=device)
    rows = torch.arange(batch_size, device=device)
    cand[rows, pos] = top_indices[pos, pick]
    return cand


class GCGOptimizer:

    def __init__(self, objective, disallowed_mask: torch.Tensor, config: dict[str, Any]):
        self.obj = objective
        self.disallowed_mask = disallowed_mask
        self.num_steps = config.get("num_steps", 500)
        self.topk = config.get("topk", 256)
        self.batch_size = config.get("batch_size", 512)
        self.check_every = config.get("check_every", 10)
        self.patience = config.get("patience", None)
        self.incremental = config.get("incremental", True)
        self.curriculum_loss = config.get("curriculum_loss", 0.5)         # active set "solved" when per-prompt loss < this
        self.curriculum_patience = config.get("curriculum_patience", 25)  # steps at a stage before force-adding
        self.save_every = config.get("save_every", None)                  # snapshot intermediate results every N steps

    def run(
        self,
        init_suffix: torch.Tensor,
        prompts: list[dict[str, Any]],
        checkpoint_fn: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        m = len(prompts)
        suffix = init_suffix.clone()
        best_loss, best_suffix = float("inf"), suffix.clone()
        mc = 1 if self.incremental else m
        stale = stage_stale = 0
        trajectory: list[dict[str, Any]] = []

        for step in range(self.num_steps):
            active = prompts[:mc]
            grad = self.obj.token_gradient(suffix, active)
            top_indices = masked_topk_per_position(grad, self.topk, self.disallowed_mask)
            cand = sample_position_swaps(suffix, top_indices, self.batch_size)
            cand = torch.cat([suffix.unsqueeze(0), cand], dim=0)    # keep current suffix, loss never increases
            losses = self.obj.batch_loss(cand, active)
            suffix = cand[int(losses.argmin())].clone()
            cur_loss = float(losses.min()) / mc

            if mc == m:
                if cur_loss < best_loss:
                    best_loss, best_suffix, stale = cur_loss, suffix.clone(), 0
                else:
                    stale += 1

            # curriculum grow: soft loss gate (active-set per-prompt loss low enough) OR step budget up
            stage_stale += 1
            added = ""
            if mc < m:
                solved = cur_loss < self.curriculum_loss
                if solved or (self.curriculum_patience and stage_stale >= self.curriculum_patience):
                    mc += 1
                    stage_stale = 0
                    added = f"  << add prompt, active={mc}/{m}" + ("" if solved else " (scheduled)")

            if step % self.check_every == 0 or step == self.num_steps - 1 or added:
                trajectory.append({"step": step, "active": mc, "loss_per_prompt": cur_loss})
                print(f"[{step:4d}] active={mc}/{m}  loss/p={cur_loss:.4f}{added}")

            if (
                checkpoint_fn is not None
                and self.save_every
                and (step + 1) % self.save_every == 0
                and step != self.num_steps - 1
            ):
                checkpoint_fn({
                    "step": step,
                    "best_suffix": best_suffix.clone(),
                    "best_loss": best_loss,
                    "cur_suffix": suffix.clone(),
                    "cur_loss": cur_loss,
                    "n_steps": step + 1,
                    "active_final": mc,
                    "trajectory": list(trajectory),
                })

            if self.patience is not None and mc == m and stale >= self.patience:
                print(f"[{step:4d}] early stop: best_loss={best_loss:.4f} stable for {self.patience} steps")
                break

        if best_loss == float("inf"):
            best_suffix = suffix.clone()
            best_loss = float(self.obj.batch_loss(best_suffix.unsqueeze(0), prompts)[0]) / m

        return {
            "best_suffix": best_suffix, 
            "best_loss": best_loss,
            "n_steps": step + 1, 
            "active_final": mc, 
            "trajectory": trajectory
        }
