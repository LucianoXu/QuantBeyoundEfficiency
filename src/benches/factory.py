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

LM_EVAL_TASKS: dict[str, str] = {
    "GSM8K": "gsm8k",
    "MMLU": "mmlu",
    "MMLU-Pro": "mmlu_pro",
    "HellaSwag": "hellaswag",
    "ARC": "arc_challenge",
    "TruthfulQA": "truthfulqa_mc2",
    "BBH": "bbh",
    "MATH": "minerva_math",
    "HumanEval": "humaneval",
    "gpqa": "gpqa_main_n_shot",
    "mgsm_direct": "mgsm_direct",
    # built-in hendrycks_ethics is unusable under datasets>=5.0
    "hendrycks_ethics": "hendrycks_ethics_local",
    "bbq": "bbq",
}

# Local lm_eval task configs shipped with this repo (e.g. the script-free hendrycks_ethics).
LM_EVAL_INCLUDE_PATH = str(Path(__file__).parent / "lm_eval_tasks")


def bench_factory(bench_args: dict[str, Any] | str | Path) -> Bench:

    print(" >> Benchmark Factory for", bench_args)

    if isinstance(bench_args, (str, Path)):
        bench_args = load_yaml_config(bench_args)

    bench_name = bench_args["bench_name"]
    if bench_name in LM_EVAL_TASKS:
        return LM_EVAL(bench_args, LM_EVAL_TASKS[bench_name])

    elif bench_name == "SORRY-Bench":
        from .lm_judge_tasks.sorry_bench import SorryBench
        return SorryBench(bench_args)

    elif bench_name == "DECEPTION-Bench":
        from .lm_judge_tasks.deception_bench import DeceptionBench
        return DeceptionBench(bench_args)

    elif bench_name == "IF-Bench":
        from .lm_judge_tasks.if_bench import IFBench
        return IFBench(bench_args)

    elif bench_name == "WILD-Bench":
        from .lm_judge_tasks.wild_bench import WildBench
        return WildBench(bench_args)

    else:
        raise ValueError("Invalid Benchmark Name: ", bench_name)    

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
        from lm_eval.tasks import TaskManager as _TaskManager
        simple_evaluate: Any = _simple_evaluate
        HFLM: Any = _HFLM

        # Make this repo's local task configs for script-free hendrycks_ethics
        task_manager: Any = _TaskManager(include_path=LM_EVAL_INCLUDE_PATH)

        log_samples = self.bench_args.get("log_samples", True)

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
            task_manager=task_manager,
            num_fewshot=self.bench_args.get("num_fewshot"),
            limit=self.bench_args.get("limit"),
            log_samples=log_samples,
            confirm_run_unsafe_code=True,   # For HumanEval to execute python code on evaluation
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
        )

        if res is None or "results" not in res:
            raise RuntimeError("lm_eval returned no results")

        summary = {k: v for k, v in res.items() if k != "samples"}
        print(" >> Saving the results to", output_dir / "result.json")
        save_json(summary, output_dir / "result.json")

        for task, records in res.get("samples", {}).items():
            samples_path = output_dir / f"samples_{task}.jsonl"
            print(" >> Saving the samples to", samples_path)
            save_jsonl(records, samples_path)

        return res
