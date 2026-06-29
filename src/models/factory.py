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

def model_factory(model_args: dict[str, Any] | str | Path) -> tuple[PreTrainedTokenizerBase, GenerationMixin]:
    print(" >> Model Factory for", model_args)

    if isinstance(model_args, (str, Path)):
        model_args = load_yaml_config(model_args)

    model_name = model_args["model_name"]

    if model_name not in SUPPORTED_MODELS: 
        raise ValueError("Invalid Model Name: ", model_name)

    return HF_standard_model_factory(model_args)
    

def HF_standard_model_factory(model_args: dict[str, Any]) -> tuple[PreTrainedTokenizerBase, GenerationMixin]:

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
        return awq_model_factory(model_args, 'W4A16_ASYM')

    else:
        raise ValueError("Invalid quantization type.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map=model_args["device_map"]
    )
    if quant_type in ["int4", "int2"]:
        model = torch.compile(model)
    return tokenizer, model
