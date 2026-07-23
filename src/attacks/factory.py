from typing import Any
from pathlib import Path

from ..utils import load_yaml_config
from .runner import run_attack


def attack_factory(args: dict[str, Any] | str | Path) -> Any:
    """ Either hands over or Loads the config arguments to the attack pipeline. """
    if isinstance(args, (str, Path)):
        args = load_yaml_config(args)
    return run_attack(args)
