from typing import Any
from pathlib import Path
import torch

from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationMixin


def awq_checkpoint_ready(save_dir: Path) -> bool:
    return save_dir.is_dir() and (save_dir / "config.json").exists() and any(save_dir.glob("*.safetensors"))


def awq_model_factory(model_args: dict[str, Any], awq_scheme: str, seed: int) -> tuple[PreTrainedTokenizerBase, GenerationMixin]:
    model_name = model_args["model_name"]
    device_map = model_args["device_map"]
    save_dir = Path(model_args["awq_save_dir"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if awq_checkpoint_ready(save_dir):
        print(" >> Found cached AWQ checkpoint at", save_dir, "-> loading")
    else:
        print(" >> No cached AWQ checkpoint at", save_dir, "-> calibrating now")
        calibrate_and_save_awq(model_name, tokenizer, save_dir, awq_scheme, seed)

    model = AutoModelForCausalLM.from_pretrained(
        save_dir,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    return tokenizer, model


def calibrate_and_save_awq(
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    save_dir: Path,
    awq_scheme: str,
    seed: int,
) -> None:
    '''
    Calibrate and svae the mode. See https://github.com/vllm-project/llm-compressor/blob/main/examples/awq/llama_example.py.
    '''

    # lazy import. llmcompressor takes very long time.
    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier

    # use fixed calibration parameters for now
    num_samples = 256
    max_seq_len = 512
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=seed).select(range(num_samples))

    def _to_text(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    def _tokenize(example):
        return tokenizer(
            example["text"],
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(_to_text, remove_columns=ds.column_names)
    ds = ds.map(_tokenize, remove_columns=ds.column_names)

    recipe = [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(ignore=["lm_head"], scheme=awq_scheme, targets=["Linear"]),
    ]

    save_dir.mkdir(parents=True, exist_ok=True)

    oneshot(
        model=model_name,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq_len,
        num_calibration_samples=num_samples,
        output_dir=str(save_dir),
    )
    tokenizer.save_pretrained(save_dir)

    torch.cuda.empty_cache()
