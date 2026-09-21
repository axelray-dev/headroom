"""Regression tests for transport errors returned by the LiteLLM backend."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.backends.base import BackendResponse  # noqa: E402
from headroom.backends.litellm import LiteLLMBackend  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

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


def _erroring_anthropic_backend() -> MagicMock:
    """Raise blank-message transport errors through both proxy backend paths."""

    async def send_message(body: dict, headers: dict) -> BackendResponse:
        raise httpx.ReadError("")

    async def stream_message(body: dict, headers: dict) -> AsyncIterator[object]:
        raise httpx.ReadTimeout("")
        yield  # pragma: no cover - keeps this function an async generator

    backend = MagicMock()
    backend.name = "anyllm-anthropic"
    backend.send_message = send_message
    backend.stream_message = stream_message
    backend.map_model_id = MagicMock(return_value="claude-3-5-sonnet-20241022")
    backend.supports_model = MagicMock(return_value=True)
    return backend


def _proxy_config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="anthropic",
    )


def _messages_request(*, stream: bool) -> dict:
    return {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
        "stream": stream,
    }


def test_anthropic_proxy_names_nonstream_transport_error_without_message() -> None:
    backend = _erroring_anthropic_backend()
    with patch("headroom.proxy.server.AnyLLMBackend", return_value=backend):
        app = create_app(_proxy_config())
        with TestClient(app) as client:
            response = client.post(
                "/v1/messages",
                json=_messages_request(stream=False),
                headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
            )

    assert response.status_code == 500
    assert response.json()["error"]["message"] == "ReadError (no message)"


def test_bedrock_stream_names_transport_error_without_message() -> None:
    backend = _erroring_anthropic_backend()
    with patch("headroom.proxy.server.AnyLLMBackend", return_value=backend):
        app = create_app(_proxy_config())
        with TestClient(app) as client:
            response = client.post(
                "/v1/messages",
                json=_messages_request(stream=True),
                headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
            )

    assert response.status_code == 200
    assert '"message": "ReadTimeout (no message)"' in response.text
