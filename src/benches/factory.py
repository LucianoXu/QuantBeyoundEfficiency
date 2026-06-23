from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod

from transformers import PreTrainedTokenizerBase
from transformers.generation.utils import GenerationMixin

from ..utils import load_yaml_config, save_jsonl, save_json


class Bench(ABC):

    def __init__(self, bench_args: dict[str, Any]):
        self.bench_args = bench_args

    @abstractmethod
    def eval(self, tokenizer: PreTrainedTokenizerBase, model: GenerationMixin,
             output_dir: Path) -> dict[str, Any]:
        ...


def bench_factory(bench_args: dict[str, Any] | str | Path) -> Bench:

    print(" >> Benchmark Factory for", bench_args)


    if isinstance(bench_args, (str, Path)):
        bench_args = load_yaml_config(bench_args)

    if bench_args["bench_name"] == "GSM8K":
        return LM_EVAL(bench_args, "gsm8k")

    else:
        raise ValueError("Invalid Benchmark Name: ", bench_args["bench_name"])
    

class LM_EVAL(Bench):
    def __init__(self, bench_args: dict[str, Any], task: str):
        super().__init__(bench_args)
        self.task = task

    def eval(self, tokenizer: PreTrainedTokenizerBase, model: GenerationMixin,
             output_dir: Path) -> dict[str, Any]:
        
        print(" >> Evaluating. Will output to", output_dir)

        # surpress false positive typing errors
        from lm_eval import simple_evaluate as _simple_evaluate
        from lm_eval.models.huggingface import HFLM as _HFLM
        simple_evaluate: Any = _simple_evaluate
        HFLM: Any = _HFLM

        log_samples = self.bench_args.get("log_samples", True)
        # single run-level seed drives all of lm_eval's internal RNGs, otherwise it
        # falls back to its own defaults (0 / 1234) and ignores our global seeding.
        seed = self.bench_args.get("random_seed", 0)

        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=self.bench_args.get("batch_size", 1),
            max_length=self.bench_args.get("max_length"),
        )
        res = simple_evaluate(
            model=lm,
            tasks=self.bench_args.get("tasks", [self.task]),
            num_fewshot=self.bench_args.get("num_fewshot"),
            limit=self.bench_args.get("limit"),
            log_samples=log_samples,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
        )

        if res is None or "results" not in res:
            raise RuntimeError("lm_eval returned no results")

        # save the benchmark results
        summary = {k: v for k, v in res.items() if k != "samples"}
        print(" >> Saving the results to", output_dir / "result.json")
        save_json(summary, output_dir / "result.json")

        # save the examples
        for task, records in res.get("samples", {}).items():
            samples_path = output_dir / f"samples_{task}.jsonl"
            print(" >> Saving the samples to", samples_path)
            save_jsonl(records, samples_path)

        return res
