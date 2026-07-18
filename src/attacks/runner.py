# runs the adversarial suffix attack

import gc
from pathlib import Path
from typing import Any

import torch

from ..utils import save_yaml_config, save_json, save_jsonl, seed_everything, tee_console, collect_environment
from ..models.factory import model_factory
from . import scaffold as S
from .objective import build_objective
from .parallel import cuda_devices, build_sharded
from .optimizer import build_optimizer
from .eval import evaluate_on_model


def run_attack(args: dict[str, Any]) -> dict[str, Any]:
    """ Runs the attack pipeline including data preparation, optimizing the adversarial suffix and evaluate the results."""

    output_dir = Path(args["output_dir"])

    output_dir.mkdir(parents=True, exist_ok=False)

    with tee_console(output_dir / "log.txt", mode="w"):
        print(" >> Adversarial-Suffix Attack Experiment")
        seed = args.get("random_seed", 0)
        seed_everything(seed)

        objective_kind = args["objective"]  # 'single' | 'differential'
        atk = args.get("attack", {})
        ev = args.get("eval", {})

        save_yaml_config(args, output_dir / "config.yaml")
        save_json(collect_environment(), output_dir / "env.json")

        # detect and use mutliple GPUs
        shard = atk.get("shard", True)
        devices = cuda_devices(atk.get("max_replicas")) if shard else cuda_devices(1)
        extra_devices = devices[1:]
        multi = len(devices) > 1
        if multi:
            print(f" >> Multi-GPU: primary {devices[0]} + {len(extra_devices)} replica(s)")

        # load full and quantized models
        model_args = dict(args["model"])
        model_args["config_type"] = "model"

        if multi and devices[0].type == "cuda":
            model_args["device_map"] = {"": devices[0].index}   # put primary in GPU0
        tokenizer, model = model_factory(model_args, seed=seed)
        model.eval()
        model.requires_grad_(False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        embedding_layer = model.get_input_embeddings()
        E = embedding_layer.weight
        device = model.device

        # build prompts and masks
        l_suf = atk.get("l_suf", 20)
        train_raw, test_raw = S.load_sorry_bench_prompts(
            atk.get("n_train", 8),
            atk.get("n_test", 8),
            atk.get("sample_seed", 0),
            prompt_style=atk.get("prompt_style", "base")
        )
        train = S.build_prompts(train_raw, tokenizer, embedding_layer, l_suf, device, atk.get("target_max_words", 20))
        test = S.build_prompts(test_raw, tokenizer, embedding_layer, l_suf, device, atk.get("target_max_words", 20))

        print(" >> Building disallowed-token mask ...")
        disallowed = S.build_disallowed_mask(tokenizer, E.shape[0], device)
        
        print(f" >> allowed tokens: {int((~disallowed).sum())}/{E.shape[0]}")
        init_suffix = S.init_suffix_ids(tokenizer, l_suf, device, atk.get("init_token", "!"))

        # build the objective
        proxy_stats = {}
        model_q = None
        if objective_kind == "differential":
            from .proxy import load_fake_quant_model
            print(" >> Loading fake-quant proxy ...")
            model_q = load_fake_quant_model(model_args["model_name"], model_args.get("device_map"))
        objective = build_objective(objective_kind, model, l_suf, atk, tokenizer, model_q)

        # share across multiple GPUs
        if multi:
            objective = build_sharded(
                objective, 
                train, 
                objective_kind, 
                model_args["model_name"],
                tokenizer, 
                train_raw, 
                l_suf, atk, 
                extra_devices
            )
            proxy_stats["n_gpus"] = len(devices)

        # build optimizer
        opt_name = atk.get("optimizer", "gcg")
        optimizer = build_optimizer(opt_name, objective, disallowed, atk)

        print(f" >> Optimizing with {opt_name} (objective={objective_kind}, m_train={len(train)}) ...")
        result = optimizer.run(init_suffix, train)

        best_suffix = result["best_suffix"].detach().cpu()
        suffix_text = tokenizer.decode(best_suffix)

        print(f"\n >> best loss/p = {result['best_loss']:.4f}")
        print(f" >> suffix = {suffix_text!r}")

        save_jsonl(result["trajectory"], output_dir / "trajectory.jsonl")
        save_json(
            {
                "suffix": suffix_text, 
                "suffix_ids": best_suffix.tolist(),
                "best_loss": result["best_loss"], 
                "n_steps": result["n_steps"],
                "active_final": result["active_final"]
            }, 
            output_dir / "suffix.json"
        )

        # evaluate on bf16 and int4, with/without suffix
        if hasattr(objective, "free"):
            objective.free()
        del objective, optimizer, model, embedding_layer, E
        if model_q is not None:
            del model_q

        gc.collect()
        torch.cuda.empty_cache()

        gen_tok = ev.get("gen_max_new_tokens", 256)
        eval_bs = ev.get("eval_batch_size", 16)
        model_name = args["model"]["model_name"]
        device_map = args["model"].get("device_map", "auto")

        results: dict[str, Any] = {
            "objective": objective_kind,
            "optimizer": atk.get("optimizer", "gcg"),
            "best_loss": result["best_loss"],
            "n_steps": result["n_steps"],
            "active_final": result["active_final"],
            "suffix": suffix_text, **proxy_stats,
        }
        all_samples: list[dict[str, Any]] = []
        
        empty_suffix = torch.empty(0, dtype=torch.long)
        conditions = {
            "attacked": best_suffix, 
            "baseline": empty_suffix
        }
        recs: dict[str, dict[str, dict[str, list]]] = {
            c: {"bf16": {}, "int4": {}} for c in conditions
        }

        for quant_type in ("bf16", "int4"):
            spec = {
                "config_type": "model",
                "model_name": model_name,
                "quant_type": quant_type,
                "device_map": device_map,
            }
            print(f" >> Loading real {quant_type} model for eval ...")
            tok, mdl = model_factory(spec, seed=seed)
            mdl.eval()
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token

            for cond, suf in conditions.items():
                for split, pset in (("train", train), ("test", test)):
                    records = evaluate_on_model(mdl, tok, pset, suf, gen_tok, eval_bs)
                    recs[cond][quant_type][split] = records
                    for rec in records:
                        all_samples.append({"model": quant_type, "condition": cond, "split": split, **rec})

            del mdl, tok
            gc.collect()
            torch.cuda.empty_cache()

        
        def split_metrics(bf, iq):
            n = len(bf)
            return {
                "n": n,
                "bf16_asr": sum(r["jailbroken"] for r in bf) / n if n else 0.0,
                "int4_asr": sum(r["jailbroken"] for r in iq) / n if n else 0.0,
                "differential": sum(q["jailbroken"] and not b["jailbroken"] for b, q in zip(bf, iq)) / n if n else 0.0,
            }

        for split in ("train", "test"):
            results[split] = {
                cond: split_metrics(recs[cond]["bf16"][split], recs[cond]["int4"][split])
                for cond in conditions
            }
            a, b = results[split]["attacked"], results[split]["baseline"]
            print(f" >> [{split}] attacked bf16/int4/diff={a['bf16_asr']:.0%}/{a['int4_asr']:.0%}/{a['differential']:.0%}"
                  f"  |  baseline={b['bf16_asr']:.0%}/{b['int4_asr']:.0%}/{b['differential']:.0%}")

        save_json(results, output_dir / "result.json")
        save_jsonl(all_samples, output_dir / "samples.jsonl")

        print(" >> Results Saved.")

    return results
