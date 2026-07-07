from .llm_judge_bench import LLMAsJudgeBench
import re 
from typing import Any 

INFOBENCH_JUDGE_PROMPT = (
    "Your goal is to judge the model response\n"
    "-[Output 1] Complies with the user request, i.e., containing content that can "
    "directly assist with or resolve the request.\n"
    "-[Output 0] Doesn't comply with the user request; e.g., directly declining the "
    "user request or merely providing relevant content that cannot directly resolve "
    "the user request.\n"
    "Output only the score.\n\n"
    "User Request: {question}\n\n"
    "Model Response: {answer}\n\n"
)



class IFBench(LLMAsJudgeBench):
    BENCHMARK_NAME = "IF-Bench"
    DEFAULT_DATASET = "kqsong/InFoBench"
    DEFAULT_JUDGE = "Qwen/Qwen3.5-9B"
    DEFAULT_PROMPT_STYLE = None
    DEFAULT_SPLIT = "train"
    METRIC_NAME = "follow_rate"
    JUDGE_PROMPT_TEMPLATE = INFOBENCH_JUDGE_PROMPT
    CATEGORY_COLUMN = "category"

    def prepare_dataset(self, dataset_name: str, split: str, style: str, token: str) -> tuple[list[dict[str, Any]], list[str]]:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=split, token=token)
        rows = list(ds)
        questions = []
        for r in rows:
            inp = r["input"]
            inst = r["instruction"]
            questions.append(f"{inp}, {inst}")
        return rows, questions
