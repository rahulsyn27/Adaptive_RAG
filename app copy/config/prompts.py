"""Prompt loading utilities."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROMPTS_PATH = Path(__file__).with_name("prompts.yaml")


@lru_cache(maxsize=1)
def load_prompts() -> dict[str, Any]:
    """Load YAML prompts shipped with the application."""
    with PROMPTS_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("prompts.yaml must contain a mapping")
    return data


def get_prompt(name: str, key: str) -> str:
    """Return a named prompt template string."""
    prompts = load_prompts()
    section = prompts.get(name)
    if not isinstance(section, dict) or key not in section:
        raise KeyError(f"Missing prompt '{name}.{key}'")
    return str(section[key])
