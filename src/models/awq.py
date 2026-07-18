from typing import Any
from pathlib import Path
import torch

from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationMixin

DEFAULT_AWQ_SAVE_DIR= Path("./awq_calibration")

def awq_checkpoint_ready(save_dir: Path) -> bool:
    """ Checks if a valid AWQ checkpoint exists in the target directory.

    Args:
        save_dir: The directory path to check for the AWQ checkpoint.

    Returns:
        True if awq checkpoint exists, otherwise False.
    """

    return save_dir.is_dir() and (save_dir / "config.json").exists() and any(save_dir.glob("*.safetensors"))


def awq_model_factory(model_args: dict[str, Any], awq_scheme: str, seed: int) -> tuple[PreTrainedTokenizerBase, GenerationMixin]:
    """ Retrieves the AWQ quantized model. Initializes calibration if no cache exists.

    Args:
        model_args: Configuration properties initialized the model setup.
        awq_scheme: The precision scheme (e.g. W4A16).
        seed: Randomized seed utilized for dataset shuffling for the calibration.

    Returns:
        A tuple containing:
            - The tokenizer for the given target model.
            - The quantized target model.
    """

    model_name = model_args["model_name"]
    device_map = model_args["device_map"]

    save_dir = model_args.get("awq_save_dir")
    if not save_dir:
        print(f"Could not find awq_save_dir for {model_name}")
        print(f"Falling back the default directory {DEFAULT_AWQ_SAVE_DIR}")
        save_dir = DEFAULT_AWQ_SAVE_DIR / model_name
    else:
        save_dir = Path(save_dir)

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
    """ Executes post-training AWQ calibration and saves the model.

    Args:
        model_name: Huggingface model identifier.
        tokenizer: Tokenizer matching the target model.
        save_dir: Save destination for the target models calibrated weights and tokenizer.
        awq_scheme: The precision scheme (e.g. W4A16).
        seed: Randomized seed utilized for dataset shuffling for the calibration.

    Note: datasets and llmcompressor components are loaded lazily.

    For more information see: https://github.com/vllm-project/llm-compressor/blob/main/examples/awq/llama_example.py.
    """



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
