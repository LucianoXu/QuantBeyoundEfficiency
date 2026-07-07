import gc
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from transformers.generation.utils import GenerationMixin

from .factory import Bench
from ..utils import save_json, save_jsonl
from ..envvar import require_hf_token
import re 

class LLMAsJudgeBench(Bench):
    BENCHMARK_NAME: str = "Base-Bench"
    DEFAULT_DATASET: str = ""
    DEFAULT_JUDGE: str = ""
    DEFAULT_PROMPT_STYLE: str = "base"
    DEFAULT_SPLIT: str = "train"
    METRIC_NAME: str = "score"
    JUDGE_PROMPT_TEMPLATE: str = ""
    
    CATEGORY_COLUMN: str | None = None 

    def prepare_dataset(self, dataset_name: str, split: str, style: str, token: str) -> tuple[list[dict[str, Any]], list[str]]:
        raise NotImplementedError

    def get_judge_prompt(self, row: dict[str, Any], question: str, response: str) -> str:
        return self.JUDGE_PROMPT_TEMPLATE.format(question=question, answer=response)

    #match first number
    def parse_score(self, verdict: str) -> float | None:
        if not verdict:
            return None
        match = re.search(r"\d+(?:\.\d+)?", verdict)
        return float(match.group()) if match else None

    def eval(self, tokenizer: PreTrainedTokenizerBase, model: GenerationMixin,
             output_dir: Path) -> dict[str, Any]:

        print(f" >> Evaluating {self.BENCHMARK_NAME}. Will output to {output_dir}")

        dataset_name = self.bench_args.get("dataset", self.DEFAULT_DATASET)
        judge_name = self.bench_args.get("judge_model", self.DEFAULT_JUDGE)
        style = self.bench_args.get("prompt_style", self.DEFAULT_PROMPT_STYLE)
        batch_size = self.bench_args.get("batch_size", 8)
        max_new_tokens = self.bench_args.get("max_new_tokens", 512)
        limit = self.bench_args.get("limit")
        log_samples = self.bench_args.get("log_samples", True)
        chat_kwargs = self.bench_args.get("chat_template_kwargs", {})

        token = require_hf_token()

        rows, questions = self.prepare_dataset(dataset_name, split=self.bench_args.get("split", self.DEFAULT_SPLIT), style=style, token=token)
        if limit:
            rows = rows[:limit]
            questions = questions[:limit]
        print(f" >> {len(rows)} prompts (prompt_style={style})")

        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True, **chat_kwargs,
            )
            for q in questions
        ]
        responses = self._generate(tokenizer, model, prompts, max_new_tokens, batch_size, desc="generating")

        judge_tokenizer = AutoTokenizer.from_pretrained(judge_name, token=token)
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_name, torch_dtype=torch.bfloat16, device_map="auto", token=token,
        )
        judge_prompts = [
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": self.get_judge_prompt(row, q, r)}],
                tokenize=False, add_generation_prompt=True, enable_thinking = False, 
            )
            for row, q, r in zip(rows, questions, responses)
        ]
        
        verdicts = self._generate(
            judge_tokenizer,
            judge_model,
            judge_prompts,
            max_new_tokens=8,
            batch_size=batch_size,
            desc="judging",
        )

        scores = [self.parse_score(v) for v in verdicts]
        n_invalid = sum(s is None for s in scores)
        if n_invalid:
            print(f" >> WARNING: {n_invalid}/{len(scores)} judge verdicts unparseable; "
                  f"excluded from {self.METRIC_NAME}")

        del judge_model, judge_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        per_category: dict[Any, list[float]] = defaultdict(list)
        records = []
        for row, q, resp, score in zip(rows, questions, responses, scores):
            category = row.get(self.CATEGORY_COLUMN) if self.CATEGORY_COLUMN else None
            
            if score is not None and category is not None:
                per_category[category].append(score)
            
            records.append({
                "question_id": row.get("question_id") or row.get("id"),
                "category": category,
                "prompt_style": style,
                "question": q,
                "response": resp,
                "score": score,
            })

        valid = [s for s in scores if s is not None]
        overall_score = sum(valid) / len(valid) if valid else 0.0

        metric_results = {
            self.METRIC_NAME: overall_score,
            "n_samples": len(valid),
            "n_invalid": n_invalid,
            "prompt_style": style,
            "judge_model": judge_name,
        }
        
        if self.CATEGORY_COLUMN:
            metric_results[f"per_category_{self.METRIC_NAME}"] = {
                cat: sum(s) / len(s) for cat, s in sorted(per_category.items())
            }

        summary = {
            "results": {
                self.BENCHMARK_NAME: metric_results
            },
            "config": {
                "dataset": dataset_name,
                "prompt_style": style,
                "judge_model": judge_name,
                "max_new_tokens": max_new_tokens,
                "limit": limit,
            },
        }

        print(" >> Saving the results to", output_dir / "result.json")
        save_json(summary, output_dir / "result.json")
        if log_samples:
            samples_filename = f"samples_{self.BENCHMARK_NAME.lower().replace('-', '_')}.jsonl"
            samples_path = output_dir / samples_filename
            print(" >> Saving the samples to", samples_path)
            save_jsonl(records, samples_path)

        return summary

    @staticmethod
    def _generate(tokenizer: PreTrainedTokenizerBase, model: GenerationMixin,
                  prompts: list[str], max_new_tokens: int, batch_size: int,
                  desc: str = "generating") -> list[str]:
        old_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        outputs = []
        for i in tqdm(range(0, len(prompts), batch_size), desc=desc, unit="batch"):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            prompt_len = enc["input_ids"].shape[1]
            for seq in gen:
                outputs.append(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))

        tokenizer.padding_side = old_side
        print("outputs: ", outputs)
        return outputs
