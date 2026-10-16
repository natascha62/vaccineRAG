"""Provider-independent LLM client interface."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from vaxrag.config import ModelConfig
from vaxrag.schemas import LLMResult


class PdfDocument(BaseModel):
    """A PDF attached to a request (arm A3)."""

    title: str
    filename: str
    data: bytes


class LLMRequest(BaseModel):
    """One request: a system prompt, a user prompt and an optional JSON schema.

    Attached PDFs are placed before the user text. With `cache_documents`, the
    clients mark them for prompt caching so that later requests sharing the
    same leading documents are billed at the cache-read price.
    """

    system: str
    user: str
    json_schema: dict[str, Any] | None = None
    schema_name: str = "response"
    images_png: list[bytes] = Field(default_factory=list)
    documents: list[PdfDocument] = Field(default_factory=list)
    cache_documents: bool = False


class LLMError(Exception):
    """A request that failed after the SDK's own retries (network, quota, 5xx, 4xx)."""


class LLMRefusal(Exception):
    """The provider declined to answer (safety refusal or content filter)."""

    def __init__(self, message: str, result: LLMResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class LLMClient(ABC):
    """Base class for all providers.

    Subclasses translate an `LLMRequest` into the provider's API call and
    return an `LLMResult` with the raw text, the provider-reported model version
    and token usage. Transient errors are retried by the provider SDKs
    (`max_retries`); anything that still fails is raised as `LLMError`.
    """

    max_retries: int = 5

    def __init__(self, key: str, config: ModelConfig) -> None:
        self.key = key
        self.config = config

    def api_key(self, default_env: str) -> str | None:
        """Read the API key from the configured (or default) environment variable."""
        return os.environ.get(self.config.api_key_env or default_env)

    def generation_settings(self) -> dict[str, Any]:
        """Return the settings recorded with every call for reproducibility."""
        return {
            "provider": self.config.provider,
            "model_id": self.config.model_id,
            "temperature": self.config.temperature,
            "reasoning_effort": self.config.reasoning_effort,
            "max_output_tokens": self.config.max_output_tokens,
        }

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResult:
        """Send one request and return the result."""

    def cost_usd(self, result: LLMResult) -> float | None:
        """Estimate the cost of a call from token usage and configured prices.

        `input_tokens` includes cached tokens; cache reads and writes are billed
        at their own prices when configured (otherwise at the input price).
        """
        if result.input_tokens is None or result.output_tokens is None:
            return None
        cfg = self.config
        read = result.cache_read_tokens or 0
        write = result.cache_write_tokens or 0
        uncached = max(result.input_tokens - read - write, 0)
        read_price = (
            cfg.price_cache_read_per_mtok
            if cfg.price_cache_read_per_mtok is not None
            else cfg.price_input_per_mtok
        )
        write_price = (
            cfg.price_cache_write_per_mtok
            if cfg.price_cache_write_per_mtok is not None
            else cfg.price_input_per_mtok
        )
        return (
            uncached * cfg.price_input_per_mtok
            + read * read_price
            + write * write_price
            + result.output_tokens * cfg.price_output_per_mtok
        ) / 1_000_000
