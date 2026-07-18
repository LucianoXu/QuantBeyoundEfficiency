import gc
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from transformers.generation.utils import GenerationMixin

from ..factory import Bench
from ...utils import save_json, save_jsonl
from ...envvar import require_hf_token
import re 

class LLMAsJudgeBench(Bench):
    """ Abstract base benchmark for evaluating benchmarks using a LLM-as-a-judge.

    This class serves as the base interface for loading the evaluation datsets that require a LLM judge.
    It includes evaluation, prompt formatting, handling batched text generation for the target model and the judge
    and computes performance metrics. Subclasses must override 'prepare_dataset'.


    Attributes:
        BENCHMARK_NAME: Standard identifier for this benchmark.
        DATASET_NAME: The huggingface repo id
        PROMPT_STYLE: Default prompt style to load. Defaults to base.
        SPLIT: Default split to get from huggingface. Defaults to train.
        METRIC_NAME: The name of the resulting evaluation metric. Defaults to score.
        JUDGE_PROMPT_TEMPLATE: String template to used for judge evaluation.
        CATEGORY_COLUMN: Column indicating the subcategory of interest.
        ID_COLUMN: Unique column identifier.
    """
    BENCHMARK_NAME: str = None
    DATASET_NAME: str = ""
    PROMPT_STYLE: str = "base"
    SPLIT: str = "train"
    METRIC_NAME: str = "score"
    JUDGE_PROMPT_TEMPLATE: str = ""
    CATEGORY_COLUMN: str = None 
    ID_COLUMN: str = ""

    def prepare_dataset(self, dataset_name: str, split: str, style: str, token: str) -> tuple[list[dict[str, Any]], list[str]]:
        """Abstract method to load and format the dataset.

        Args:
            dataset_name: Huggingface ID
            split: The split to load.
            style: Styling to pull from the dataset.
            token: Huggingface user authentication token for faster loading times or to access gated datasets.

        Returns:
            A tuple containing two lists:
            - A list of dictionaries holding the full row data of the dataset.
            - A list of pre-formatted string questions ready to evaluate.

        Raises:
            NotImplementedError: Needs to be implemented by the corresponding subclass.
        """
        raise NotImplementedError

    def get_judge_prompt(self, row: dict[str, Any], question: str, response: str) -> str:
        """ Formats the input prompt for evaluated by the LLM judge.

         Args:
             row: Dictionary tracking the data of a given row sample.
             question: The specific question that the target model had to answer.
             response: The generated answer from the target model for the given question.

        Returns:
            The prompt string given to the judge model to evaluate the response.
         """
        return self.JUDGE_PROMPT_TEMPLATE.format(question=question, answer=response)

    def parse_score(self, verdict: str) -> float | None:
        """ Parse the numerical verdict (0.0 or 1.0) from raw judge text output.

        Args:
            verdict: The raw string text response from the judge model.

        Returns:
            A float (0.0 or 1.0) if parsing is successful otherwise None.
        """
        if not verdict:
            return None
        numbers = re.findall(r"\d+(?:\.\d+)?", verdict.strip())
        if not numbers:
            return None  
        score = float(numbers[-1])
        if score in (0.0, 1.0):
            return score
        return None

    def eval(self, tokenizer: PreTrainedTokenizerBase, model: GenerationMixin,
             output_dir: Path) -> dict[str, Any]:
        """ Runs the full evaluation pipeline including answer generation, judge scoring and logging results.

        Args:
            tokenizer: The Huggingface tokenizer matching the target model.
            model: The target model that gets evaluated.
            output_dir: The directory path to dump the results and generated samples in json format.

        Returns:
            A nested configuration summary dictionary containing:
            - 'results': The results containing the over all main benchmark score metrics with additional infos.
            - 'config': An overview of the run configuration (e.g. dataset, max_new_tokens, limit)
        """

        print(f" >> Evaluating {self.BENCHMARK_NAME}. Will output to {output_dir}")
        # get config arguments
        style = self.bench_args.get("prompt_style", self.PROMPT_STYLE)
        batch_size = self.bench_args["batch_size"]
        max_new_tokens = self.bench_args["max_new_tokens"]
        limit = self.bench_args["limit"]
        log_samples = self.bench_args["log_samples"]
        chat_kwargs = self.bench_args["chat_template_kwargs"]
        judge_args = self.bench_args["judge_args"]
        judge_name = judge_args["name"]
        judge_enable_thinking = judge_args["enable_thinking"]
        judge_gen_len = judge_args["gen_len"]
        judge_batch_size =judge_args["batch_size"]
        token = require_hf_token()

        rows, questions = self.prepare_dataset(self.DATASET_NAME, split=self.bench_args.get("split", self.SPLIT), style=style, token=token)
        if limit:
            rows = rows[:limit]
            questions = questions[:limit]
        print(f" >> {len(rows)} prompts (prompt_style={style})")

        # Token truncation for overly long questions
        max_q_tokens = self.bench_args.get("max_question_tokens")
        if max_q_tokens:
            n_before = sum(len(tokenizer(q, add_special_tokens=False)["input_ids"]) > max_q_tokens for q in questions)
            questions = [
                tokenizer.decode(
                    tokenizer(q, add_special_tokens=False, truncation=True, max_length=max_q_tokens)["input_ids"]
                )
                for q in questions
            ]
            print(f" >> truncated {n_before}/{len(questions)} questions to {max_q_tokens} tokens")

        # Generate model response
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True, **chat_kwargs,
            )
            for q in questions
        ]
        responses = self._generate(tokenizer, model, prompts, max_new_tokens, batch_size, desc="generating")

        # Load judge and score outputs
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_name, token=token)
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_name, torch_dtype=torch.bfloat16, device_map="auto", token=token,
        )
        judge_prompts = [
            judge_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are an LLM Judge, your goal is to judge the Model Response, you only output one number: 1 or 0. You are forbidden from outputting any text. "},
                    {"role": "user", "content": self.get_judge_prompt(row, q, r)}
                ],
                tokenize=False, add_generation_prompt=True, enable_thinking=judge_enable_thinking, 
            )
            for row, q, r in zip(rows, questions, responses)
        ]
        
        verdicts = self._generate(
        judge_tokenizer, judge_model, judge_prompts, judge_gen_len, judge_batch_size,
        enable_thinking=judge_enable_thinking, desc="judging",
        )   

        scores = [self.parse_score(v) for v in verdicts]
        n_invalid = sum(s is None for s in scores)
        if n_invalid:
            print(f" >> WARNING: {n_invalid}/{len(scores)} judge verdicts unparseable; "
                  f"excluded from {self.METRIC_NAME}")

        # Free GPU memory from judge model
        del judge_model, judge_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # Compute category metrics and save results
        per_category: dict[Any, list[float]] = defaultdict(list)
        records = []
        for idx, (row, q, resp, score) in enumerate(zip(rows, questions, responses, scores)):
            category = row.get(self.CATEGORY_COLUMN) if self.CATEGORY_COLUMN else None
            
            if score is not None and category is not None:
                per_category[category].append(score)
            
            question_id = row.get(self.ID_COLUMN) if (self.ID_COLUMN in row) else f"{idx}"
            
            records.append({
                "question_id": question_id,
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
                "dataset": self.DATASET_NAME,
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
    def _generate(tokenizer: PreTrainedTokenizerBase, model: GenerationMixin, prompts: list[str] , max_new_tokens: int, batch_size: int,
                enable_thinking: bool = False, desc: str = "generating") -> list[str]:
        """ Helper utility for greedy batched generation

        Args:
            tokenizer: The Huggingface tokenizer matching the target model.
            model: The target model that gets evaluated.
            prompts: A list of the given prompts
            max_new_tokens: Limit cap to the length of the generation
            batch_size: Total amount of prompt rows feed per iteration
            enable_thinking: If True, extends the token limit by 512 to allow for longer thinking generation.
                Defaults to False.
            desc: Custom description label for the tqdm loading bar. Defaults to 'generating'

        Returns:
            A list of processed response strings stripped of their initial prompt
        """
        
        old_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        outputs = []
        if enable_thinking:
            max_new_tokens += 512
        for i in tqdm(range(0, len(prompts), batch_size), desc=desc, unit="batch"):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,pad_token_id=tokenizer.pad_token_id)
            prompt_len = enc["input_ids"].shape[1]
            for seq in gen:
                outputs.append(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))
        print("Outputs: ", outputs)
        tokenizer.padding_side = old_side
        return outputs
