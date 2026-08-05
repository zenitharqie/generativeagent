import pytest

from src.llm import (
    ActionOutput,
    EmptyLLMOutputError,
    LLMOutputParseError,
    LLMOutputValidationError,
    PlaceholderLLMOutputError,
    parse_action_output,
    parse_importance_output,
    parse_plan_output,
    parse_reflection_output,
)


def test_parsers_accept_documented_structured_outputs() -> None:
    assert parse_plan_output('{"plan":"Ask a concise question."}').plan == (
        "Ask a concise question."
    )
    assert parse_action_output(
        '{"response":"I can help.","memory_to_store":"Monday offered help.",'
        '"category":"social_learning"}'
    ) == ActionOutput(
        response="I can help.",
        memory_to_store="Monday offered help.",
        category="social_learning",
    )
    reflection = parse_reflection_output(
        '{"reflection":"Clear roles support the group.",'
        '"category":"peer_interaction"}'
    )
    assert reflection.reflection == "Clear roles support the group."
    assert parse_importance_output(
        '{"importance":7,"reason":"It may affect later participation."}'
    ).importance == 7.0


def test_parser_performs_only_minimal_outer_fence_cleanup() -> None:
    output = parse_plan_output('  ```json\n{"plan":"Respond briefly."}\n```  ')
    assert output.plan == "Respond briefly."


@pytest.mark.parametrize("raw_output", ["", "   ", "```json\n\n```"])
def test_parser_rejects_empty_output(raw_output: str) -> None:
    with pytest.raises(EmptyLLMOutputError):
        parse_action_output(raw_output)


@pytest.mark.parametrize(
    "raw_output",
    [
        "not JSON",
        '["response", "memory"]',
        "```python\n{}\n```",
        "```json\n{}",
    ],
)
def test_parser_rejects_malformed_or_non_object_output(raw_output: str) -> None:
    with pytest.raises(LLMOutputParseError):
        parse_action_output(raw_output)


@pytest.mark.parametrize(
    "raw_output",
    [
        '{"memory_to_store":"Stored.","category":"classroom_norms"}',
        '{"response":"Spoken.","category":"classroom_norms"}',
        '{"response":"Spoken.","memory_to_store":"Stored."}',
        '{"response":"","memory_to_store":"Stored.",'
        '"category":"classroom_norms"}',
    ],
)
def test_action_parser_rejects_missing_or_empty_required_fields(
    raw_output: str,
) -> None:
    with pytest.raises(LLMOutputValidationError):
        parse_action_output(raw_output)


@pytest.mark.parametrize(
    "raw_output",
    [
        '{"plan":"short plan summary"}',
        '{"plan":"{{ generated_plan }}"}',
    ],
)
def test_parser_rejects_unmistakable_placeholder_values(raw_output: str) -> None:
    with pytest.raises(PlaceholderLLMOutputError):
        parse_plan_output(raw_output)


@pytest.mark.parametrize("importance", [True, 0, 11, float("inf"), "7"])
def test_importance_parser_rejects_invalid_scores(importance: object) -> None:
    raw_output = (
        '{"importance":'
        + (f'"{importance}"' if isinstance(importance, str) else str(importance).lower())
        + ',"reason":"Reason."}'
    )
    with pytest.raises((LLMOutputParseError, LLMOutputValidationError)):
        parse_importance_output(raw_output)
