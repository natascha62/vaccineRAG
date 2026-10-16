"""Create LLM clients from configuration keys."""

from __future__ import annotations

from vaxrag.config import Config
from vaxrag.llm.base import LLMClient


def create_client(config: Config, model_key: str) -> LLMClient:
    """Instantiate the client for `config.models[model_key]`.

    SDK imports happen lazily so that, e.g., running only open-source models
    does not require every provider's credentials.
    """
    if model_key not in config.models:
        raise KeyError(f"Model '{model_key}' is not defined in the config")
    model_cfg = config.models[model_key]
    if model_cfg.provider == "anthropic":
        from vaxrag.llm.anthropic_client import AnthropicClient

        return AnthropicClient(model_key, model_cfg)
    if model_cfg.provider == "openai":
        from vaxrag.llm.openai_client import OpenAICompatibleClient

        return OpenAICompatibleClient(model_key, model_cfg)
    if model_cfg.provider == "vllm":
        from vaxrag.llm.openai_client import VLLMClient

        return VLLMClient(model_key, model_cfg)
    if model_cfg.provider == "google":
        from vaxrag.llm.gemini_client import GeminiClient

        return GeminiClient(model_key, model_cfg)
    raise ValueError(f"Unknown provider: {model_cfg.provider}")
