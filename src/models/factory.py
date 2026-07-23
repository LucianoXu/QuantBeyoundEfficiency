from typing import Any
from pathlib import Path
import torch

from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, QuantoConfig
from transformers.generation.utils import GenerationMixin

from ..utils import load_yaml_config

from .awq import awq_model_factory

SUPPORTED_MODELS = {
    "Qwen/Qwen3-4B", 
    "Qwen/Qwen3-1.7B", 
}

def model_factory(model_args: dict[str, Any] | str | Path, seed: int = None) -> tuple[PreTrainedTokenizerBase, GenerationMixin]:
    """ Builds and dispatches language models based in input arguments loaded from the config .yaml

    Args:
        model_args: Model arguments from the loaded .yaml config or a string showing to the .yaml config
        seed: Randomization seed if AWQ quantization method is chosen. Defaults to None.

    Returns:
        A tuple containing two lists:
            - The tokenizer matching the target model
            - The configured and instiated target model

    Raises:
        ValueError: If the provided model does not map to any registered model
    """
    print(" >> Model Factory for", model_args)

    if isinstance(model_args, (str, Path)):
        model_args = load_yaml_config(model_args)

    model_name = model_args["model_name"]

    if model_name not in SUPPORTED_MODELS: 
        raise ValueError("Invalid Model Name: ", model_name)

    return HF_standard_model_factory(model_args, seed=seed)
    

def HF_standard_model_factory(model_args: dict[str, Any], seed: int = None) -> tuple[PreTrainedTokenizerBase, GenerationMixin]:
    """ Laods a Huggingface model and optionally applies the quantization from the model_args.

    Args:
        model_args: Configuration properties loaded from the .yaml file. Must include 'model_name',
        'quant_type' and 'device_map'.
        seed: Randomization seed if AWQ quantization method is chosen. Defaults to None.


    Returns:
    A tuple containing two lists:
        - The tokenizer matching the target model
        - The configured and instiated target model

    Raises:
        ValueError: If the provided quantization/precision does not map to any registered quantization/precision.
    """
    model_name = model_args["model_name"]
    quant_type = model_args["quant_type"]

    if quant_type == "bf16":
        quant_config = None

    elif quant_type == "int8":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    elif quant_type == "nf4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quant_type == "int4": 
        quant_config = QuantoConfig(weights='int4')
    elif quant_type == "int2": 
        quant_config = QuantoConfig(weights='int2')

    elif quant_type == "awq-w4a16-asym":
        return awq_model_factory(model_args, 'W4A16_ASYM',seed=seed)

    else:
        raise ValueError("Invalid quantization type.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map=model_args["device_map"]
    )
    return tokenizer, model
