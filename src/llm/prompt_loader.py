"""Strict loading, identity, versioning, and hashing for production prompts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from types import MappingProxyType

from src.llm.errors import PromptIntegrityError, PromptLoadError, PromptRenderError


DEFAULT_PROMPT_DIRECTORY = Path(__file__).resolve().parents[2] / "prompts"
PROMPT_VERSION = "phase10_pilot_v1"
_PLACEHOLDER_PATTERN = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


@dataclass(frozen=True)
class PromptSpec:
    template_id: str
    filename: str
    version: str
    expected_placeholders: tuple[str, ...]
    output_contract: str
    model_role: str


@dataclass(frozen=True)
class PromptTemplate:
    spec: PromptSpec
    content: str
    content_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, PromptSpec):
            raise PromptIntegrityError("prompt spec must be PromptSpec")
        if not isinstance(self.content, str) or not self.content.strip():
            raise PromptIntegrityError("prompt content must not be empty")
        if re.fullmatch(r"[0-9a-f]{64}", self.content_hash) is None:
            raise PromptIntegrityError("prompt content_hash must be SHA-256 hex")

    def render_trusted(self, replacements: Mapping[str, str]) -> str:
        if not isinstance(replacements, Mapping):
            raise PromptRenderError("trusted prompt replacements must be a mapping")
        if set(replacements) != set(self.spec.expected_placeholders):
            raise PromptRenderError(
                "trusted prompt replacement keys do not match the template contract"
            )
        rendered = self.content
        for name in self.spec.expected_placeholders:
            value = replacements[name]
            if not isinstance(value, str) or not value.strip():
                raise PromptRenderError(f"trusted replacement {name!r} must not be empty")
            rendered = rendered.replace("{{" + name + "}}", value)
        if _PLACEHOLDER_PATTERN.search(rendered) is not None:
            raise PromptRenderError("rendered trusted prompt contains an unresolved placeholder")
        return rendered


_SPECS = {
    "plan": PromptSpec(
        "plan", "plan.txt", PROMPT_VERSION, ("OUTPUT_SCHEMA",), "plan", "simulator"
    ),
    "natural_interaction": PromptSpec(
        "natural_interaction",
        "natural_interaction.txt",
        PROMPT_VERSION,
        ("OUTPUT_SCHEMA",),
        "action",
        "simulator",
    ),
    "target_event_response": PromptSpec(
        "target_event_response",
        "target_event_response.txt",
        PROMPT_VERSION,
        ("OUTPUT_SCHEMA",),
        "action",
        "simulator",
    ),
    "reflect": PromptSpec(
        "reflect", "reflect.txt", PROMPT_VERSION, ("OUTPUT_SCHEMA",), "reflection", "simulator"
    ),
    "importance_scoring": PromptSpec(
        "importance_scoring",
        "importance_scoring.txt",
        PROMPT_VERSION,
        ("OUTPUT_SCHEMA",),
        "importance",
        "simulator",
    ),
    "bcr_judge": PromptSpec(
        "bcr_judge",
        "bcr_judge.txt",
        PROMPT_VERSION,
        ("OUTPUT_SCHEMA",),
        "bcr_judgment",
        "evaluator",
    ),
}
PROMPT_SPECS: Mapping[str, PromptSpec] = MappingProxyType(_SPECS)


class PromptLoader:
    """Load only the exact Phase 10 production prompt inventory."""

    def __init__(self, prompt_directory: str | Path = DEFAULT_PROMPT_DIRECTORY) -> None:
        self._directory = Path(prompt_directory)

    @property
    def prompt_directory(self) -> Path:
        return self._directory

    def validate_inventory(self) -> None:
        try:
            files = {
                path.name
                for path in self._directory.iterdir()
                if path.is_file() and not path.name.startswith(".")
            }
        except OSError as exc:
            raise PromptLoadError(
                f"could not inspect prompt directory {self._directory}"
            ) from exc
        expected = {spec.filename for spec in PROMPT_SPECS.values()}
        if files != expected:
            raise PromptIntegrityError(
                "production prompt inventory does not match the six-file contract"
            )

    def load(self, template_id: str) -> PromptTemplate:
        try:
            spec = PROMPT_SPECS[template_id]
        except (KeyError, TypeError) as exc:
            raise PromptLoadError(f"unsupported prompt template ID: {template_id!r}") from exc
        path = self._directory / spec.filename
        try:
            if path.is_symlink():
                raise PromptLoadError(f"prompt file must not be a symlink: {spec.filename}")
            payload = path.read_bytes()
        except PromptLoadError:
            raise
        except OSError as exc:
            raise PromptLoadError(f"could not load prompt file {spec.filename}") from exc
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PromptLoadError(f"prompt file is not UTF-8: {spec.filename}") from exc
        if not content.strip():
            raise PromptIntegrityError(f"prompt file is empty: {spec.filename}")
        if "\r" in content:
            raise PromptIntegrityError(f"prompt file must use LF line endings: {spec.filename}")
        placeholders = tuple(_PLACEHOLDER_PATTERN.findall(content))
        if placeholders != spec.expected_placeholders:
            raise PromptIntegrityError(
                f"prompt placeholders do not match contract for {spec.filename}"
            )
        scrubbed = _PLACEHOLDER_PATTERN.sub("", content)
        if "{{" in scrubbed or "}}" in scrubbed:
            raise PromptIntegrityError(
                f"prompt contains malformed placeholder braces: {spec.filename}"
            )
        return PromptTemplate(
            spec=spec,
            content=content,
            content_hash=sha256(payload).hexdigest(),
        )

    def load_all(self) -> tuple[PromptTemplate, ...]:
        self.validate_inventory()
        return tuple(self.load(template_id) for template_id in PROMPT_SPECS)


__all__ = [
    "DEFAULT_PROMPT_DIRECTORY",
    "PROMPT_SPECS",
    "PROMPT_VERSION",
    "PromptLoader",
    "PromptSpec",
    "PromptTemplate",
]
