"""Additive, sanitized, offline-tested Phase 10 smoke CLI."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import main as main_module
from src.llm import LLMSmokeTestResult
from src.llm.errors import ProviderError
from src.llm.provider import StructuredTextRequest, StructuredTextResponse
from src.llm.smoke import run_real_llm_smoke_test
from src.llm.usage import ProviderUsage


class SmokeProvider:
    def __init__(self) -> None:
        self.requests: list[StructuredTextRequest] = []

    def generate_structured(self, request: StructuredTextRequest) -> StructuredTextResponse:
        self.requests.append(request)
        return StructuredTextResponse(
            '{"plan":"GENERATED-CONTENT-MUST-NOT-BE-PRINTED"}',
            ProviderUsage(8, 4, 12),
        )


def test_smoke_helper_makes_one_minimal_call_and_discards_content() -> None:
    provider = SmokeProvider()

    result = run_real_llm_smoke_test(provider)

    assert len(provider.requests) == 1
    assert result == LLMSmokeTestResult(
        provider="openai",
        model_identity="openai:gpt-5-nano-2025-08-07",
        attempt_count=1,
        input_tokens=8,
        output_tokens=4,
        total_tokens=12,
    )
    assert "GENERATED-CONTENT" not in "\n".join(result.sanitized_lines())


def test_smoke_cli_success_prints_only_sanitized_status_and_usage(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "sk-secret-never-print"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    result = LLMSmokeTestResult(
        provider="openai",
        model_identity="openai:gpt-5-nano-2025-08-07",
        attempt_count=1,
        input_tokens=8,
        output_tokens=4,
        total_tokens=12,
    )
    monkeypatch.setattr("src.llm.run_real_llm_smoke_test", lambda: result)

    assert main_module.main(["llm-smoke-test", "--real"]) == 0
    captured = capsys.readouterr()

    assert captured.out.splitlines() == list(result.sanitized_lines())
    assert captured.err == ""
    assert secret not in captured.out


def test_missing_api_key_fails_before_sdk_client_creation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    created = False

    def forbidden_factory(key: str, timeout: int) -> SimpleNamespace:
        nonlocal created
        created = True
        raise AssertionError("client creation must not occur")

    monkeypatch.setattr("src.llm.openai_client._default_client_factory", forbidden_factory)

    assert main_module.main(["llm-smoke-test", "--real"]) == 1
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == "LLM smoke test failed: CredentialError\n"
    assert created is False


def test_smoke_cli_failure_never_prints_raw_provider_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "phase10-cli-provider-secret"

    def failed_smoke() -> LLMSmokeTestResult:
        raise ProviderError(sentinel)

    monkeypatch.setattr("src.llm.run_real_llm_smoke_test", failed_smoke)

    assert main_module.main(["llm-smoke-test", "--real"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "LLM smoke test failed: ProviderError\n"
    assert sentinel not in captured.err


def test_smoke_cli_requires_explicit_real_flag_and_rejects_overrides(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as missing:
        main_module.main(["llm-smoke-test"])
    assert missing.value.code == 2
    capsys.readouterr()

    with pytest.raises(SystemExit) as override:
        main_module.main(
            ["llm-smoke-test", "--real", "--model", "gpt-5-nano"]
        )
    assert override.value.code == 2


def test_existing_cli_modes_remain_present() -> None:
    help_text = main_module.build_parser().format_help()

    assert "validate-config" in help_text
    assert "run-pilot" in help_text
    assert "evaluate" in help_text
    assert "llm-smoke-test" in help_text
