"""Request-building tests for provider clients (no network calls)."""

from types import SimpleNamespace

import pytest

from vaxrag.config import Config, ModelConfig
from vaxrag.llm.base import LLMError, LLMRefusal, LLMRequest
from vaxrag.llm.factory import create_client
from vaxrag.schemas import LLMResult

SCHEMA = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


@pytest.fixture(autouse=True)
def fake_keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(var, "test-key")


def _client(provider, **kw):
    cfg = Config.model_validate({"models": {"m": {"provider": provider, "model_id": "x", **kw}}})
    return create_client(cfg, "m")


def test_anthropic_params_use_output_config_and_omit_temperature():
    client = _client("anthropic", temperature=None, reasoning_effort="low", max_output_tokens=123)
    params = client.build_params(LLMRequest(system="s", user="u", json_schema=SCHEMA))
    assert params["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert params["system"] == "s" and params["max_tokens"] == 123
    assert params["messages"][0]["content"][-1] == {"type": "text", "text": "u"}
    assert "extra_body" not in params and "temperature" not in params
    with_temp = _client("anthropic", temperature=0.0).build_params(LLMRequest(system="s", user="u"))
    assert with_temp["extra_body"] == {"temperature": 0.0}


def test_anthropic_refusal_and_errors(monkeypatch):
    client = _client("anthropic")
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="")],
        model="claude-x",
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="bio"),
        usage=SimpleNamespace(input_tokens=5, output_tokens=0),
    )
    monkeypatch.setattr(client.client.messages, "create", lambda **kw: response)
    with pytest.raises(LLMRefusal, match="bio"):
        client.generate(LLMRequest(system="s", user="u"))

    import anthropic
    import httpx2 as httpx_mod  # anthropic 1.x uses httpx2

    def boom(**kw):
        raise anthropic.APIConnectionError(request=httpx_mod.Request("POST", "https://x"))

    monkeypatch.setattr(client.client.messages, "create", boom)
    with pytest.raises(LLMError):
        client.generate(LLMRequest(system="s", user="u"))


def test_openai_params_strict_schema_and_reasoning():
    client = _client("openai", temperature=None, reasoning_effort="low", max_output_tokens=50)
    params = client.build_params(
        LLMRequest(system="s", user="u", json_schema=SCHEMA, schema_name="ans")
    )
    assert params["response_format"]["json_schema"] == {
        "name": "ans",
        "schema": SCHEMA,
        "strict": True,
    }
    assert params["reasoning_effort"] == "low" and "temperature" not in params
    assert params["max_completion_tokens"] == 50
    assert params["messages"][0] == {"role": "system", "content": "s"}


def test_vllm_params_drop_reasoning_and_use_max_tokens():
    client = _client(
        "vllm", temperature=0.0, reasoning_effort="low", base_url="http://localhost:8000/v1"
    )
    params = client.build_params(LLMRequest(system="s", user="u", json_schema=SCHEMA))
    assert "reasoning_effort" not in params and "max_completion_tokens" not in params
    assert params["max_tokens"] == 4096 and params["temperature"] == 0.0
    assert client.generation_settings()["inference_framework"] == "vllm"


def test_openai_generate_maps_usage(monkeypatch):
    client = _client("openai")
    response = SimpleNamespace(
        model="gpt-x-2026",
        choices=[
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content='{"a":1}', refusal=None)
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
    )
    monkeypatch.setattr(client.client.chat.completions, "create", lambda **kw: response)
    result = client.generate(LLMRequest(system="s", user="u"))
    assert (
        result.text,
        result.model_version,
        result.input_tokens,
        result.output_tokens,
        result.reasoning_tokens,
    ) == ('{"a":1}', "gpt-x-2026", 10, 4, 2)


def test_gemini_config():
    client = _client("google", temperature=None, reasoning_effort="low")
    cfg = client.build_config(LLMRequest(system="s", user="u", json_schema=SCHEMA))
    assert cfg.system_instruction == "s"
    assert cfg.response_mime_type == "application/json" and cfg.response_json_schema == SCHEMA
    assert cfg.thinking_config.thinking_level is not None
    assert cfg.temperature is None


def test_cost_estimate():
    client = _client("vllm", price_input_per_mtok=2.0, price_output_per_mtok=10.0)
    assert client.cost_usd(
        LLMResult(text="", input_tokens=1_000_000, output_tokens=100_000)
    ) == pytest.approx(3.0)
    assert client.cost_usd(LLMResult(text="")) is None


def test_factory_rejects_unknown_model():
    with pytest.raises(KeyError):
        create_client(Config(models={"a": ModelConfig(provider="vllm", model_id="x")}), "b")
