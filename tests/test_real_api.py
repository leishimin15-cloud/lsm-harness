"""Integration tests using the real DeepSeek API.

These tests use ``deepseek-v4-flash`` (the cheapest model) to verify
the full end-to-end pipeline.  They are SKIPPED unless explicitly
enabled, even when an API key is configured.

Run with:  LSM_RUN_REAL_API=1 pytest tests/test_real_api.py -v -s
"""

from __future__ import annotations

import os
import time

import pytest

from lsm_harness.config import Settings
from lsm_harness.ai.providers import get_client
from lsm_harness.ai.types import ModelResponse, ToolCall, Usage


# ── helpers ───────────────────────────────────────────────────────

def _has_api_key() -> bool:
    """Check whether paid/network integration tests were explicitly enabled."""
    enabled = os.getenv("LSM_RUN_REAL_API", "").lower() in {"1", "true", "yes"}
    if not enabled:
        return False
    key = (
        os.getenv("LSM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or ""
    ).strip()
    return bool(key) and key not in {"your-key-here", "replace-me"}


def _make_client():
    """Build a client using the flash model (cheapest)."""
    return get_client(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        small_model="deepseek-v4-flash",
        thinking="disabled",
        timeout=60.0,
    )


# ── basic connectivity tests ──────────────────────────────────────


@pytest.mark.skipif(not _has_api_key(), reason="No DeepSeek API key configured")
class TestRealAPIBasics:
    """Smoke tests that verify the API is reachable and responds."""

    def test_simple_reply(self):
        """Flash model returns a non-empty reply."""
        client = _make_client()
        resp = client.complete(
            model="deepseek-v4-flash",
            system="回复'OK'即可，不要其他内容。",
            messages=[{"role": "user", "content": "ping"}],
            tools=[],
            max_tokens=20,
        )
        assert isinstance(resp, ModelResponse)
        assert resp.text is not None
        assert len(resp.text) > 0
        assert resp.stop_reason == "stop"
        assert resp.usage.input_tokens > 0
        assert resp.usage.output_tokens > 0

    def test_streaming_reply(self):
        """Streaming returns text deltas and a final done event."""
        client = _make_client()
        deltas = list(
            client.stream_complete(
                model="deepseek-v4-flash",
                system="回复'OK'。",
                messages=[{"role": "user", "content": "ping"}],
                tools=[],
                max_tokens=20,
            )
        )
        assert len(deltas) >= 2  # at least text_delta + done
        kinds = [d.kind for d in deltas]
        assert "text_delta" in kinds
        assert "done" in kinds

    def test_tool_calling(self):
        """Model calls the correct tool when instructed."""
        client = _make_client()
        resp = client.complete(
            model="deepseek-v4-flash",
            system="你必须调用 echo 工具，参数 value 设为 'hello'。不要回复文字。",
            messages=[{"role": "user", "content": "echo hello"}],
            tools=[{
                "name": "echo",
                "description": "Echo a value back",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            }],
            max_tokens=100,
        )
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) >= 1
        assert resp.tool_calls[0].name == "echo"
        assert "hello" in str(resp.tool_calls[0].arguments)

    def test_chinese_reply(self):
        """Model responds in Chinese when asked in Chinese."""
        client = _make_client()
        resp = client.complete(
            model="deepseek-v4-flash",
            system="你是中文助手。回答要简洁。",
            messages=[{"role": "user", "content": "你好，你是谁？"}],
            tools=[],
            max_tokens=50,
        )
        # Should contain Chinese characters
        assert any("\u4e00" <= c <= "\u9fff" for c in resp.text)
        assert len(resp.text) > 0


# ── tool result processing tests ──────────────────────────────────


@pytest.mark.skipif(not _has_api_key(), reason="No DeepSeek API key configured")
class TestRealAPIToolProcessing:
    """Tests that verify tool results are processed correctly."""

    def test_tool_result_handling(self):
        """Model processes tool results and generates a follow-up reply."""
        client = _make_client()
        tools = [{
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }]

        # Turn 1: model calls the tool
        resp1 = client.complete(
            model="deepseek-v4-flash",
            system="你必须调用 get_weather 工具查北京天气。不要回复文字。",
            messages=[{"role": "user", "content": "北京天气怎么样？"}],
            tools=tools,
            max_tokens=100,
        )
        assert len(resp1.tool_calls) >= 1

        # Turn 2: provide tool result, get summary
        resp2 = client.complete(
            model="deepseek-v4-flash",
            system="你是天气助手。用中文回复。",
            messages=[
                {"role": "user", "content": "北京天气怎么样？"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": resp1.tool_calls[0].id,
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "北京"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": resp1.tool_calls[0].id,
                    "content": "北京今天晴天，25°C。",
                },
            ],
            tools=tools,
            max_tokens=100,
        )
        assert len(resp2.text) > 0
        assert "25" in resp2.text or "晴" in resp2.text or "天气" in resp2.text


# ── performance test ──────────────────────────────────────────────


@pytest.mark.skipif(not _has_api_key(), reason="No DeepSeek API key configured")
class TestRealAPIPerformance:
    """Basic latency checks."""

    def test_response_latency(self):
        """Flash model responds within a reasonable time."""
        client = _make_client()
        start = time.monotonic()
        resp = client.complete(
            model="deepseek-v4-flash",
            system="回复'OK'。",
            messages=[{"role": "user", "content": "ping"}],
            tools=[],
            max_tokens=10,
        )
        elapsed = time.monotonic() - start
        # Flash model should respond within 15 seconds
        assert elapsed < 15.0, f"Response took {elapsed:.1f}s, expected <15s"
        assert len(resp.text) > 0

    def test_streaming_latency_to_first_token(self):
        """Streaming delivers the first token quickly."""
        client = _make_client()
        start = time.monotonic()
        first_token_time = None
        for delta in client.stream_complete(
            model="deepseek-v4-flash",
            system="回复一个简短的问候。",
            messages=[{"role": "user", "content": "你好"}],
            tools=[],
            max_tokens=30,
        ):
            if delta.kind == "text_delta" and first_token_time is None:
                first_token_time = time.monotonic() - start
        assert first_token_time is not None, "No text delta received"
        # First token should arrive within 10 seconds
        assert first_token_time < 10.0, (
            f"TTFT: {first_token_time:.1f}s, expected <10s"
        )


# ── error handling test ───────────────────────────────────────────


@pytest.mark.skipif(not _has_api_key(), reason="No DeepSeek API key configured")
class TestRealAPIErrors:
    """Tests that verify error handling works."""

    def test_invalid_model_rejected(self):
        """Calling a non-existent model raises an error."""
        client = _make_client()
        with pytest.raises(Exception):
            client.complete(
                model="nonexistent-model-xyz",
                system="",
                messages=[{"role": "user", "content": "test"}],
                tools=[],
                max_tokens=10,
            )
