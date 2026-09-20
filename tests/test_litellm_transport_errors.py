"""Regression tests for transport errors returned by the LiteLLM backend."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.backends.litellm import LiteLLMBackend  # noqa: E402

_BODY = {
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "hello"}],
}


@pytest.mark.asyncio
async def test_send_message_names_transport_error_without_message() -> None:
    with (
        patch(
            "headroom.backends.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=httpx.ReadError(""),
        ),
        patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
    ):
        backend = LiteLLMBackend(provider="bedrock", region="us-east-1")
        result = await backend.send_message(_BODY, {})

    assert result.status_code == 500
    assert result.error == "ReadError (no message)"
    assert result.body["error"]["message"] == "ReadError (no message)"


@pytest.mark.asyncio
async def test_stream_message_names_transport_error_without_message() -> None:
    with (
        patch(
            "headroom.backends.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=httpx.ReadTimeout(""),
        ),
        patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
    ):
        backend = LiteLLMBackend(provider="bedrock", region="us-east-1")
        events = [event async for event in backend.stream_message(_BODY, {})]

    error_event = next(event for event in events if event.event_type == "error")
    assert error_event.data["error"]["message"] == "ReadTimeout (no message)"
