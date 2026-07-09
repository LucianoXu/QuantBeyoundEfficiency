# Single-Node Multi-GPU training support. Code assisted by AI.

import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from .objective import build_objective
from . import scaffold as S


def cuda_devices(max_replicas: int | None = None) -> list[torch.device]:
    if not torch.cuda.is_available():
        return [torch.device("cpu")]
    n = torch.cuda.device_count()
    if max_replicas:
        n = min(n, max_replicas)
    return [torch.device(f"cuda:{i}") for i in range(n)]


def _dev_ctx(dev: torch.device):
    return torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()


class ShardedObjective:

    def __init__(self, objectives: list, per_device_prompts: list[list[dict[str, Any]]],
                 devices: list[torch.device]):
        assert len(objectives) == len(per_device_prompts) == len(devices)
        self.objectives = objectives
        self.per_device_prompts = per_device_prompts
        self.devices = devices
        self.primary = objectives[0]
        self.E, self.l_suf, self.V = self.primary.E, self.primary.l_suf, self.primary.V
        self._pool = ThreadPoolExecutor(max_workers=len(objectives))

    # gradient + curriculum bookkeeping run on the primary device
    def token_gradient(self, suffix_ids, active):
        mc = len(active)
        return self.primary.token_gradient(suffix_ids.to(self.devices[0]), self.per_device_prompts[0][:mc])

    def diagnostics(self, suffix_ids, active):
        mc = len(active)
        return self.primary.diagnostics(suffix_ids.to(self.devices[0]), self.per_device_prompts[0][:mc])

    def batch_loss(self, cand, active):
        mc = len(active)
        shards = torch.chunk(cand, len(self.objectives), dim=0)     # contiguous, order-preserving

        def work(i: int, shard: torch.Tensor):
            with _dev_ctx(self.devices[i]):
                l = self.objectives[i].batch_loss(shard.to(self.devices[i]), self.per_device_prompts[i][:mc])
            return l.to(self.devices[0])

        futs = {self._pool.submit(work, i, s): i for i, s in enumerate(shards)}
        parts: list[Any] = [None] * len(shards)
        for f in futs:
            parts[futs[f]] = f.result()
        return torch.cat(parts, dim=0)

    def free(self) -> None:
        self._pool.shutdown(wait=True)
        # drop extra replicas (device 0 is freed by the runner alongside the primary models)
        for obj in self.objectives[1:]:
            for attr in ("model", "model_full", "model_q"):
                if hasattr(obj, attr):
                    setattr(obj, attr, None)
        self.objectives = self.objectives[:1]


def build_sharded(primary_objective, primary_prompts, kind: str, model_name: str, tokenizer,
                  train_raw: list[dict[str, Any]], l_suf: int, atk: dict[str, Any],
                  extra_devices: list[torch.device], dtype=torch.bfloat16) -> ShardedObjective:
    
    from transformers import AutoModelForCausalLM
    from .proxy import load_fake_quant_model

    objectives = [primary_objective]
    prompts = [primary_prompts]
    devices = [primary_objective.E.device]

    for dev in extra_devices:

        print(f" >> Loading replica on {dev} ...")
        idx = dev.index if dev.type == "cuda" else None
        dmap = {"": idx} if idx is not None else None
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=dmap).eval()

        if dev.type != "cuda":
            model = model.to(dev)
        model.requires_grad_(False)
        model_q = None

        if kind == "differential":
            model_q = load_fake_quant_model(model_name, dmap, dtype)
            if dev.type != "cuda":
                model_q = model_q.to(dev)

        rep_prompts = S.build_prompts(
            train_raw, 
            tokenizer, 
            model.get_input_embeddings(), 
            l_suf, 
            dev,
            atk.get("target_max_words", 20)
        )

        objectives.append(build_objective(kind, model, l_suf, atk, tokenizer, model_q))
        prompts.append(rep_prompts)
        devices.append(dev)

    return ShardedObjective(objectives, prompts, devices)
