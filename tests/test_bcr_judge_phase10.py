"""Condition-blind BCR evaluator service and strict judged-value tests."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import pytest

from src.evaluation import BCRModelMetadata, build_bcr_export, load_evaluation_run
from src.evaluation.errors import (
    BCRJudgmentError,
    BCRJudgmentFailure,
    BCRJudgmentIndependenceError,
)
from src.evaluation.judgment_reader import load_bcr_export
from src.evaluation.judgment_service import (
    BCRJudgeService,
    judge_bcr_export,
    validate_evaluator_independence,
)
from src.llm.errors import ProviderRateLimitError
from src.llm.provider import StructuredTextRequest, StructuredTextResponse
from src.llm.usage import ProviderUsage
from phase10_test_support import (
    bcr_judge_input,
    prepare_phase09_evaluation,
)


VALID_JUDGMENT = json.dumps(
    {
        "communication_style_assessment": "consistent",
        "focus_consistency_assessment": "consistent",
        "temporal_consistency_assessment": "consistent",
        "consistent": 1,
        "reason": "The response remains consistent with the persona and prior behavior.",
    },
    sort_keys=True,
    separators=(",", ":"),
)
VALID_INCONSISTENT_JUDGMENT = json.dumps(
    {
        "communication_style_assessment": "consistent",
        "focus_consistency_assessment": "consistent",
        "temporal_consistency_assessment": "inconsistent",
        "consistent": 0,
        "reason": "A substantive temporal contradiction makes the response inconsistent.",
    },
    sort_keys=True,
    separators=(",", ":"),
)


class RepeatingJudgeProvider:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = deque(outcomes or [])
        self.requests: list[StructuredTextRequest] = []

    def generate_structured(self, request: StructuredTextRequest) -> StructuredTextResponse:
        self.requests.append(request)
        outcome = self.outcomes.popleft() if self.outcomes else VALID_JUDGMENT
        if isinstance(outcome, BaseException):
            raise outcome
        return StructuredTextResponse(str(outcome), ProviderUsage(100, 50, 150))


def test_judge_service_sends_only_exact_phase09_neutral_judge_input(
    phase08_full_artifacts,
) -> None:
    export = build_bcr_export(
        load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)
    )[0]
    provider = RepeatingJudgeProvider()

    outcome = BCRJudgeService(provider, sleeper=lambda _: None).judge(export)

    request = provider.requests[0]
    assert json.loads(request.user_data) == export.judge_input.to_dict()
    assert outcome.judgment.consistent == 1
    assert request.model == "gpt-5-2025-08-07"
    assert request.reasoning_effort == "medium"
    assert request.max_output_tokens == 700
    assert request.timeout_seconds == 60
    assert request.stream is False
    assert len(outcome.provider_attempts) == 1
    assert outcome.provider_attempts[0].attempt_index == 1
    assert outcome.provider_attempts[0].status == "success"


def test_judge_provider_payload_excludes_outer_condition_and_mra_metadata(
    phase08_full_artifacts,
) -> None:
    export = build_bcr_export(
        load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)
    )[0]
    provider = RepeatingJudgeProvider()

    BCRJudgeService(provider, sleeper=lambda _: None).judge(export)
    payload = provider.requests[0].user_data.casefold()

    for forbidden in (
        "retrieval_config_id",
        "repetition_index",
        "retrieval_weights",
        "raw_recency",
        "normalized_recency",
        "final_score",
        "mra",
        "ground_truth",
        "expected_categories",
        "memory_id",
        '"category"',
    ):
        assert forbidden not in payload


@pytest.mark.parametrize("raw_output", [VALID_JUDGMENT, VALID_INCONSISTENT_JUDGMENT])
def test_judge_accepts_exact_binary_zero_and_one(
    phase08_full_artifacts,
    raw_output: str,
) -> None:
    export = build_bcr_export(
        load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)
    )[0]

    outcome = BCRJudgeService(
        RepeatingJudgeProvider([raw_output]), sleeper=lambda _: None
    ).judge(export)

    assert outcome.judgment.consistent == json.loads(raw_output)["consistent"]


def _invalid_judgments() -> list[str]:
    base = json.loads(VALID_JUDGMENT)
    values = ["", "not json", "{}"]
    for invalid_consistent in (-1, 2, 5, 1.0, True, "1"):
        changed = dict(base)
        changed["consistent"] = invalid_consistent
        values.append(json.dumps(changed))
    changed = dict(base)
    changed["communication_style_assessment"] = "mixed"
    values.append(json.dumps(changed))
    changed = dict(base)
    changed.pop("focus_consistency_assessment")
    values.append(json.dumps(changed))
    changed = dict(base)
    changed["reason"] = ""
    values.append(json.dumps(changed))
    changed = dict(base)
    changed["overall_decision"] = "pass"
    values.append(json.dumps(changed))
    changed = dict(base)
    changed["communication_style_score"] = 5
    values.append(json.dumps(changed))
    values.append(
        VALID_JUDGMENT.replace('"consistent":1', '"consistent":1,"consistent":0')
    )
    return values


@pytest.mark.parametrize("invalid", _invalid_judgments())
def test_judge_rejects_malformed_or_out_of_contract_output_without_repair(
    phase08_full_artifacts,
    invalid: str,
) -> None:
    export = build_bcr_export(
        load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)
    )[0]
    provider = RepeatingJudgeProvider([invalid, invalid])

    with pytest.raises(BCRJudgmentError, match="invalid structured output twice"):
        BCRJudgeService(provider, sleeper=lambda _: None).judge(export)
    assert len(provider.requests) == 2


def test_judge_retries_one_transient_failure_with_same_logical_identity(
    phase08_full_artifacts,
) -> None:
    export = build_bcr_export(
        load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)
    )[0]
    delays: list[float] = []
    provider = RepeatingJudgeProvider([ProviderRateLimitError("sanitized"), VALID_JUDGMENT])

    outcome = BCRJudgeService(provider, sleeper=delays.append).judge(export)

    assert delays == [1]
    assert tuple(item.status for item in outcome.provider_attempts) == ("failed", "success")
    assert tuple(item.attempt_index for item in outcome.provider_attempts) == (1, 2)
    assert len({item.logical_call_id for item in outcome.provider_attempts}) == 1


def test_terminal_provider_failure_exposes_every_sanitized_attempt(
    phase08_full_artifacts,
) -> None:
    export = build_bcr_export(
        load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)
    )[0]
    sentinel = "phase10-terminal-provider-secret"
    provider = RepeatingJudgeProvider(
        [
            ProviderRateLimitError(sentinel),
            ProviderRateLimitError(sentinel),
        ]
    )

    with pytest.raises(BCRJudgmentFailure, match="evaluator call failed") as caught:
        BCRJudgeService(provider, sleeper=lambda _: None).judge(export)

    attempts = caught.value.provider_attempts
    assert len(provider.requests) == 2
    assert tuple(item.status for item in attempts) == ("failed", "failed")
    assert tuple(item.attempt_index for item in attempts) == (1, 2)
    assert len({item.logical_call_id for item in attempts}) == 1
    assert all(item.error_type == "ProviderRateLimitError" for item in attempts)
    serialized = json.dumps([item.to_dict() for item in attempts], sort_keys=True)
    assert sentinel not in serialized
    assert "system_instructions" not in serialized
    assert "user_data" not in serialized
    assert "output_text" not in serialized
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    with pytest.raises(AttributeError):
        caught.value.provider_attempts = ()  # type: ignore[misc]


def test_terminal_invalid_output_exposes_every_sanitized_attempt(
    phase08_full_artifacts,
) -> None:
    export = build_bcr_export(
        load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)
    )[0]
    sentinel = "phase10-terminal-invalid-output-secret"
    provider = RepeatingJudgeProvider([sentinel, sentinel])

    with pytest.raises(BCRJudgmentFailure, match="invalid structured output twice") as caught:
        BCRJudgeService(provider, sleeper=lambda _: None).judge(export)

    attempts = caught.value.provider_attempts
    assert len(provider.requests) == 2
    assert tuple(item.status for item in attempts) == (
        "invalid_output",
        "invalid_output",
    )
    assert tuple(item.attempt_index for item in attempts) == (1, 2)
    assert len({item.logical_call_id for item in attempts}) == 1
    assert all(item.error_type == "BCRJudgmentError" for item in attempts)
    serialized = json.dumps([item.to_dict() for item in attempts], sort_keys=True)
    assert sentinel not in serialized
    assert "system_instructions" not in serialized
    assert "user_data" not in serialized
    assert "output_text" not in serialized
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_primary_independence_preflight_requires_explicit_approved_distinct_models(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    missing_dir = prepare_phase09_evaluation(
        tmp_path / "missing",
        phase08_full_artifacts[0].run_paths.run_dir,
        simulator_model_identity=None,
    )
    equal_dir = prepare_phase09_evaluation(
        tmp_path / "equal",
        phase08_full_artifacts[0].run_paths.run_dir,
        simulator_model_identity="openai:gpt-5-2025-08-07",
    )

    with pytest.raises(BCRJudgmentIndependenceError, match="explicit simulator"):
        validate_evaluator_independence(load_bcr_export(missing_dir))
    with pytest.raises(BCRJudgmentIndependenceError, match="must differ"):
        validate_evaluator_independence(load_bcr_export(equal_dir))


def test_independence_failure_happens_before_any_judge_call(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
        simulator_model_identity="openai:gpt-5-2025-08-07",
    )
    provider = RepeatingJudgeProvider()
    output = tmp_path / "judged"

    with pytest.raises(BCRJudgmentIndependenceError):
        judge_bcr_export(
            evaluation_dir=evaluation,
            output_dir=output,
            provider=provider,
            sleeper=lambda _: None,
        )
    assert provider.requests == []
    assert not output.exists()
