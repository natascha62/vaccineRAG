"""OpenAI-compatible clients: OpenAI itself and self-hosted vLLM servers.

vLLM exposes an OpenAI-compatible `/v1/chat/completions` endpoint and supports
`response_format={"type": "json_schema", ...}` via guided decoding, so Gemma and
MedGemma run through exactly the same code path as the OpenAI flagship model.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import openai

from vaxrag.llm.base import LLMClient, LLMError, LLMRefusal, LLMRequest
from vaxrag.schemas import LLMResult


class OpenAICompatibleClient(LLMClient):
    """Client for OpenAI Chat Completions (and compatible servers)."""

    default_key_env = "OPENAI_API_KEY"

    def __init__(self, key: str, config: Any) -> None:
        super().__init__(key, config)
        self.client = openai.OpenAI(
            api_key=self.api_key(self.default_key_env) or "EMPTY",
            base_url=config.base_url,
            max_retries=self.max_retries,
            timeout=config.timeout_s,
        )

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        """Build keyword arguments for `chat.completions.create`."""
        if request.images_png or request.documents:
            # Documents come first so that requests sharing them share a cacheable
            # prefix (OpenAI caches prompt prefixes automatically).
            user_content: Any = [
                {
                    "type": "file",
                    "file": {
                        "filename": doc.filename,
                        "file_data": "data:application/pdf;base64,"
                        + base64.b64encode(doc.data).decode("ascii"),
                    },
                }
                for doc in request.documents
            ]
            user_content += [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(img).decode("ascii")
                    },
                }
                for img in request.images_png
            ]
            user_content.append({"type": "text", "text": request.user})
        else:
            user_content = request.user
        params: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": user_content},
            ],
            "max_completion_tokens": self.config.max_output_tokens,
        }
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.reasoning_effort:
            params["reasoning_effort"] = self.config.reasoning_effort
        if request.json_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "schema": request.json_schema,
                    "strict": True,
                },
            }
        return params

    def generate(self, request: LLMRequest) -> LLMResult:
        """Call Chat Completions and return the message text."""
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**self.build_params(request))
        except openai.APIStatusError as exc:
            raise LLMError(f"{self.config.provider} API error {exc.status_code}: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"{self.config.provider} connection error: {exc}") from exc
        latency = time.perf_counter() - start
        choice = response.choices[0]
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        prompt_details = getattr(usage, "prompt_tokens_details", None) if usage else None
        result = LLMResult(
            text=choice.message.content or "",
            model_version=response.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
            cache_read_tokens=(
                getattr(prompt_details, "cached_tokens", None) if prompt_details else None
            ),
            latency_s=latency,
            finish_reason=choice.finish_reason,
        )
        refusal = getattr(choice.message, "refusal", None)
        if refusal or choice.finish_reason == "content_filter":
            raise LLMRefusal(f"Refusal: {refusal or 'content_filter'}", result)
        return result


class VLLMClient(OpenAICompatibleClient):
    """Client for a self-hosted vLLM server (OpenAI-compatible API).

    `max_tokens` is used instead of `max_completion_tokens` for compatibility
    with older vLLM versions, and reasoning effort is never sent.
    """

    default_key_env = "VLLM_API_KEY"

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        """Adapt the OpenAI parameters for vLLM (which cannot read PDF files)."""
        if request.documents:
            raise ValueError("vLLM models cannot read PDF documents (arm A3)")
        params = super().build_params(request)
        params["max_tokens"] = params.pop("max_completion_tokens")
        params.pop("reasoning_effort", None)
        return params

    def generation_settings(self) -> dict[str, Any]:
        """Also record the serving endpoint and inference framework."""
        settings = super().generation_settings()
        settings["inference_framework"] = "vllm"
        settings["base_url"] = self.config.base_url
        return settings
