"""
Huggingface environment variable instiation and token verifcation utility.
"""

import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=False)

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN or HF_TOKEN == "hf_replace_with_your_token":
    print(
        " >> WARNING from QuantBeyoundEfficiency: "
        " >> HF_TOKEN is not set. Copy .env.example to .env and add your "
        " >> Hugging Face token (https://huggingface.co/settings/tokens)."
    )


def require_hf_token() -> str:
    """Checks if the Huggingface was instiated in .env and returns it

    Returns:
        The Huggingface token saved in the .env file.

    Raises:
        RuntimeError: If the token is missing or matches the default configuration.
    """
    if not HF_TOKEN or HF_TOKEN == "hf_replace_with_your_token":
        raise RuntimeError(
            "HF_TOKEN is not set. Copy .env.example to .env and add your "
            "Hugging Face token (https://huggingface.co/settings/tokens)."
        )
    return HF_TOKEN
