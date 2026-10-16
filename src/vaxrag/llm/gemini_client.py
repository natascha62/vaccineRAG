"""Google Gemini client using the official `google-genai` SDK.

Thinking depth is set with `thinking_level` (e.g. "low"); structured output uses
`response_mime_type="application/json"` with `response_json_schema`. Billed output
tokens include thinking tokens (`thoughts_token_count`).
"""

from __future__ import annotations

import time
from typing import Any

from google import genai
from google.genai import errors, types

from vaxrag.llm.base import LLMClient, LLMError, LLMRefusal, LLMRequest
from vaxrag.schemas import LLMResult

_BLOCKED_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


class GeminiClient(LLMClient):
    """Client for the Gemini API (API key) or Vertex AI (when GOOGLE_GENAI_USE_VERTEXAI=1)."""

    def __init__(self, key: str, config: Any) -> None:
        super().__init__(key, config)
        api_key = self.api_key("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def build_config(self, request: LLMRequest) -> types.GenerateContentConfig:
        """Build the generation config (separated for testing)."""
        kwargs: dict[str, Any] = {
            "system_instruction": request.system,
            "max_output_tokens": self.config.max_output_tokens,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.reasoning_effort:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.config.reasoning_effort
            )
        if request.json_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = request.json_schema
        return types.GenerateContentConfig(**kwargs)

    def build_contents(self, request: LLMRequest) -> list[Any]:
        """PDFs first (implicit prefix caching), then images, then the user text."""
        contents: list[Any] = [
            types.Part.from_bytes(data=doc.data, mime_type="application/pdf")
            for doc in request.documents
        ]
        contents += [
            types.Part.from_bytes(data=img, mime_type="image/png") for img in request.images_png
        ]
        contents.append(request.user)
        return contents

    def generate(self, request: LLMRequest) -> LLMResult:
        """Call `generate_content` and return the response text."""
        contents = self.build_contents(request)
        start = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.config.model_id, contents=contents, config=self.build_config(request)
            )
        except errors.APIError as exc:
            raise LLMError(f"Gemini API error {getattr(exc, 'code', '')}: {exc}") from exc
        latency = time.perf_counter() - start
        usage = response.usage_metadata
        thoughts = (usage.thoughts_token_count or 0) if usage else 0
        candidates = (usage.candidates_token_count or 0) if usage else 0
        finish = None
        if response.candidates:
            reason = response.candidates[0].finish_reason
            finish = getattr(reason, "name", None) or (str(reason) if reason else None)
        result = LLMResult(
            text=response.text or "",
            model_version=response.model_version or self.config.model_id,
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=(candidates + thoughts) if usage else None,
            reasoning_tokens=thoughts if usage else None,
            cache_read_tokens=(usage.cached_content_token_count or 0) if usage else None,
            latency_s=latency,
            finish_reason=finish,
        )
        blocked = response.prompt_feedback is not None and response.prompt_feedback.block_reason
        if blocked or finish in _BLOCKED_FINISH_REASONS:
            raise LLMRefusal(f"Gemini blocked the response (finish={finish})", result)
        return result
