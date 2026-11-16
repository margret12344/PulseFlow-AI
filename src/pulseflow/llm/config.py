from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LLM_CONFIG_PATH = PROJECT_ROOT / "configs" / "llm.yaml"


class LLMSettings(BaseModel):
    """Validated configuration for the LLM provider."""

    provider: Literal["ollama"] = "ollama"
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout: int = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0)


def load_llm_config(
    config_path: Path = LLM_CONFIG_PATH,
) -> LLMSettings:
    """Load and validate LLM settings from a YAML file."""

    if not config_path.is_file():
        raise FileNotFoundError(
            f"LLM configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict) or "llm" not in config:
        raise ValueError("Invalid LLM configuration structure.")

    try:
        return LLMSettings.model_validate(config["llm"])
    except ValidationError as exc:
        raise ValueError("Invalid LLM configuration values.") from exc