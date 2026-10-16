"""Anthropic (Claude) client using the official `anthropic` SDK.

Model-dependent settings (all driven by the config):

* `reasoning_effort` is sent as `output_config.effort`; when it is null no
  effort is sent. The `thinking` parameter is never sent, so Claude 4.6 models
  answer without thinking (thinking cannot be disabled on Claude Opus 5.5,
  which controls depth with effort only).
* Python SDK 1.x removed `temperature` from the method signature, so a
  non-null `temperature` is forwarded through `extra_body`. Claude 4.6 accepts
  it; Claude 4.7 and later reject non-default sampling parameters (use null).
* Structured output uses `output_config.format` with a JSON schema.
* PDFs (arm A3) are sent as base64 `document` blocks before the question; with
  caching enabled each block carries `cache_control` (at most 4 breakpoints are
  allowed per request, so arm A3 uses at most 3 documents).
* Server-side refusal fallbacks are deliberately NOT enabled: a fallback would
  silently answer with a different model and contaminate the comparison.
  Refusals are recorded as such instead.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import anthropic

from vaxrag.llm.base import LLMClient, LLMError, LLMRefusal, LLMRequest
from vaxrag.schemas import LLMResult


class AnthropicClient(LLMClient):
    """Client for the Anthropic Messages API."""

    def __init__(self, key: str, config: Any) -> None:
        super().__init__(key, config)
        self.client = anthropic.Anthropic(
            api_key=self.api_key("ANTHROPIC_API_KEY"),
            max_retries=self.max_retries,
            timeout=config.timeout_s,
        )

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        """Build keyword arguments for `messages.create` (separated for testing)."""
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(img).decode("ascii"),
                },
            }
            for img in request.images_png
        ]
        for doc in request.documents:
            block: dict[str, Any] = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(doc.data).decode("ascii"),
                },
                "title": doc.title,
            }
            if request.cache_documents:
                # One breakpoint per document: a later request that starts with the
                # same documents (in the same order) reads them from the cache.
                block["cache_control"] = {"type": "ephemeral"}
            content.append(block)
        content.append({"type": "text", "text": request.user})
        output_config: dict[str, Any] = {}
        if self.config.reasoning_effort:
            output_config["effort"] = self.config.reasoning_effort
        if request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        params: dict[str, Any] = {
            "model": self.config.model_id,
            "max_tokens": self.config.max_output_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": content}],
        }
        if output_config:
            params["output_config"] = output_config
        if self.config.temperature is not None:
            params["extra_body"] = {"temperature": self.config.temperature}
        return params

    def generate(self, request: LLMRequest) -> LLMResult:
        """Call the Messages API and return the text of the final answer."""
        start = time.perf_counter()
        try:
            response = self.client.messages.create(**self.build_params(request))
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Anthropic connection error: {exc}") from exc
        latency = time.perf_counter() - start
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", None) or 0
        result = LLMResult(
            text=text,
            model_version=response.model,
            # The API reports uncached input separately from cache reads/writes.
            input_tokens=usage.input_tokens + cache_read + cache_write,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            latency_s=latency,
            finish_reason=response.stop_reason,
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusal(f"Refusal (category={category})", result)
        return result
