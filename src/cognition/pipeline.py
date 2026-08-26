"""Deterministic Phase 06 cognitive orchestration over existing memory/retrieval APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any

from src.cognition.actor import Actor
from src.cognition.perception import Perception, build_perception
from src.cognition.planner import Planner
from src.cognition.query_builder import RetrievalQuery, build_retrieval_query
from src.cognition.reflector import Reflector
from src.config_loader import (
    ExperimentConfig,
    validate_experiment_config,
)
from src.domain import (
    AgentProfile,
    MemoryEntry,
    NaturalInteractionBlockStep,
    RetrievalConfig,
    SimulationStep,
    TargetEvent,
    TargetEventStep,
)
from src.domain.agents import FIXED_AGENT_IDS
from src.domain.simulation import STEP_TYPES
from src.llm.base import (
    LLMCallType,
    LLMClient,
    LLMContext,
    LLMRequest,
    TextEmbeddingProvider,
)
from src.llm.output_parsers import (
    ActionOutput,
    ImportanceOutput,
    PlanOutput,
    ReflectionOutput,
    parse_importance_output,
)
from src.memory import MemoryFactory, MemoryStream
from src.retrieval import (
    RetrievalEngine,
    RetrievalResult,
    RetrievalScoringError,
    validate_vector,
)


class CognitivePipelineError(ValueError):
    """Base error for invalid Phase 06 orchestration."""


class CognitiveInputError(CognitivePipelineError):
    """Raised when a turn references invalid or mismatched runtime input."""


class CognitiveResultValidationError(CognitivePipelineError):
    """Raised when a dependency returns an inconsistent auditable result."""


class CognitiveStateError(CognitivePipelineError):
    """Raised when the memory stream changes outside the mutation boundary."""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CognitiveInputError(f"{field_name} must be a non-empty string")
    return value.strip()


def _normalize_context(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise CognitiveInputError("conversation_context must be a tuple of strings")
    return tuple(_require_text(item, "conversation_context item") for item in value)


@dataclass(frozen=True)
class CognitiveTurnInput:
    """All caller-controlled data needed to execute one agent turn."""

    agent_id: str
    source_step_id: str
    turn_id: str
    retrieval_config_id: str
    memory_stream: MemoryStream
    logical_timestamp: int
    query_embedding: Sequence[float] | None
    conversation_context: tuple[str, ...] = ()
    event_id: str | None = None
    previous_same_agent_response: str | None = None
    reflection_requested: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "agent_id",
            "source_step_id",
            "turn_id",
            "retrieval_config_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), field_name),
            )
        if self.agent_id not in FIXED_AGENT_IDS:
            raise CognitiveInputError(f"unsupported agent_id: {self.agent_id!r}")
        if not isinstance(self.memory_stream, MemoryStream):
            raise CognitiveInputError("memory_stream must be a MemoryStream")
        if (
            isinstance(self.logical_timestamp, bool)
            or not isinstance(self.logical_timestamp, int)
            or self.logical_timestamp < 0
        ):
            raise CognitiveInputError("logical_timestamp must be a non-negative integer")
        query_embedding = validate_vector(
            self.query_embedding,
            field_name="query_embedding",
        )
        if not any(value != 0.0 for value in query_embedding):
            raise RetrievalScoringError("query_embedding must be non-zero")
        object.__setattr__(self, "query_embedding", query_embedding)
        object.__setattr__(
            self,
            "conversation_context",
            _normalize_context(self.conversation_context),
        )
        if self.event_id is not None:
            object.__setattr__(
                self,
                "event_id",
                _require_text(self.event_id, "event_id"),
            )
        if self.previous_same_agent_response is not None:
            object.__setattr__(
                self,
                "previous_same_agent_response",
                _require_text(
                    self.previous_same_agent_response,
                    "previous_same_agent_response",
                ),
            )
        if not isinstance(self.reflection_requested, bool):
            raise CognitiveInputError("reflection_requested must be boolean")


@dataclass(frozen=True)
class CognitiveResult:
    """Immutable turn output retaining retrieval and cognitive audit data."""

    turn_id: str
    agent_id: str
    source_step_id: str
    source_step_type: str
    event_id: str | None
    retrieval_config_id: str
    logical_timestamp: int
    perception: Perception
    retrieval_query: RetrievalQuery
    retrieval_result: RetrievalResult
    selected_memory_ids: tuple[str, ...]
    plan: PlanOutput
    action: ActionOutput
    reflection: ReflectionOutput | None
    reflection_requested: bool
    memory_generation_enabled: bool
    action_importance: ImportanceOutput | None
    reflection_importance: ImportanceOutput | None
    created_memories: tuple[MemoryEntry, ...]
    accessed_memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "turn_id",
            "agent_id",
            "source_step_id",
            "retrieval_config_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise CognitiveResultValidationError(
                    f"{field_name} must be a non-empty string"
                )
        if self.agent_id not in FIXED_AGENT_IDS:
            raise CognitiveResultValidationError(
                f"unsupported result agent_id: {self.agent_id!r}"
            )
        if self.source_step_type not in STEP_TYPES:
            raise CognitiveResultValidationError(
                f"unsupported result source_step_type: {self.source_step_type!r}"
            )
        if (
            isinstance(self.logical_timestamp, bool)
            or not isinstance(self.logical_timestamp, int)
            or self.logical_timestamp < 0
        ):
            raise CognitiveResultValidationError(
                "logical_timestamp must be a non-negative integer"
            )
        if not isinstance(self.perception, Perception):
            raise CognitiveResultValidationError("perception must be a Perception")
        if not isinstance(self.retrieval_query, RetrievalQuery):
            raise CognitiveResultValidationError(
                "retrieval_query must be a RetrievalQuery"
            )
        if not isinstance(self.retrieval_result, RetrievalResult):
            raise CognitiveResultValidationError(
                "retrieval_result must be a RetrievalResult"
            )
        if not isinstance(self.plan, PlanOutput):
            raise CognitiveResultValidationError("plan must be a PlanOutput")
        if not isinstance(self.action, ActionOutput):
            raise CognitiveResultValidationError("action must be an ActionOutput")
        if self.reflection is not None and not isinstance(
            self.reflection, ReflectionOutput
        ):
            raise CognitiveResultValidationError(
                "reflection must be a ReflectionOutput or None"
            )
        if not isinstance(self.reflection_requested, bool):
            raise CognitiveResultValidationError("reflection_requested must be boolean")
        if not isinstance(self.memory_generation_enabled, bool):
            raise CognitiveResultValidationError(
                "memory_generation_enabled must be boolean"
            )

        identity = (
            self.perception.agent_id,
            self.perception.source_step_id,
            self.perception.source_step_type,
            self.perception.logical_timestamp,
        )
        if identity != (
            self.agent_id,
            self.source_step_id,
            self.source_step_type,
            self.logical_timestamp,
        ):
            raise CognitiveResultValidationError(
                "perception identity does not match cognitive result"
            )
        if self.event_id != self.perception.event_id:
            raise CognitiveResultValidationError(
                "event identity does not match perception"
            )
        if self.retrieval_query.text != self.retrieval_result.query:
            raise CognitiveResultValidationError(
                "retrieval query does not match RetrievalResult"
            )
        if (
            self.retrieval_result.agent_id != self.agent_id
            or self.retrieval_result.retrieval_config_id
            != self.retrieval_config_id
        ):
            raise CognitiveResultValidationError(
                "RetrievalResult identity does not match cognitive result"
            )
        if self.selected_memory_ids != self.retrieval_result.selected_memory_ids:
            raise CognitiveResultValidationError(
                "selected_memory_ids must preserve RetrievalResult order"
            )
        if self.accessed_memory_ids != self.selected_memory_ids:
            raise CognitiveResultValidationError(
                "accessed_memory_ids must equal the selected Top-K IDs"
            )
        if self.reflection_requested != (self.reflection is not None):
            raise CognitiveResultValidationError(
                "reflection output must match the explicit reflection request"
            )
        if self.memory_generation_enabled != self.perception.memory_generation_enabled:
            raise CognitiveResultValidationError(
                "memory generation flag does not match perception"
            )

        if not isinstance(self.created_memories, tuple) or any(
            not isinstance(memory, MemoryEntry) for memory in self.created_memories
        ):
            raise CognitiveResultValidationError(
                "created_memories must be a tuple of MemoryEntry values"
            )
        expected_count = (
            0
            if not self.memory_generation_enabled
            else 1 + int(self.reflection is not None)
        )
        if len(self.created_memories) != expected_count:
            raise CognitiveResultValidationError(
                "created memory count does not match executed cognitive stages"
            )
        if not self.memory_generation_enabled:
            if self.action_importance is not None or self.reflection_importance is not None:
                raise CognitiveResultValidationError(
                    "importance output requires enabled memory generation"
                )
            return

        if not isinstance(self.action_importance, ImportanceOutput):
            raise CognitiveResultValidationError(
                "enabled action memory requires importance output"
            )
        action_memory = self.created_memories[0]
        self._validate_created_memory(action_memory, expected_type="action")
        if (
            action_memory.text != self.action.memory_to_store
            or action_memory.category != self.action.category
            or action_memory.importance != self.action_importance.importance
        ):
            raise CognitiveResultValidationError(
                "action memory does not match parsed action output"
            )

        if self.reflection is None:
            if self.reflection_importance is not None:
                raise CognitiveResultValidationError(
                    "reflection importance requires a reflection output"
                )
        else:
            if not isinstance(self.reflection_importance, ImportanceOutput):
                raise CognitiveResultValidationError(
                    "stored reflection requires importance output"
                )
            reflection_memory = self.created_memories[1]
            self._validate_created_memory(
                reflection_memory,
                expected_type="reflection",
            )
            if (
                reflection_memory.text != self.reflection.reflection
                or reflection_memory.category != self.reflection.category
                or reflection_memory.importance
                != self.reflection_importance.importance
            ):
                raise CognitiveResultValidationError(
                    "reflection memory does not match parsed reflection output"
                )

    def _validate_created_memory(
        self,
        memory: MemoryEntry,
        *,
        expected_type: str,
    ) -> None:
        if (
            memory.agent_id != self.agent_id
            or memory.memory_type != expected_type
            or memory.source_step_id != self.source_step_id
            or memory.source_step_type != self.source_step_type
            or memory.created_at != self.logical_timestamp
            or memory.last_accessed_at != self.logical_timestamp
            or memory.embedding is None
        ):
            raise CognitiveResultValidationError(
                f"created {expected_type} memory has inconsistent audit fields"
            )

    @property
    def observation(self) -> str:
        """Return the auditable perception text."""
        return self.perception.observation_text

    @property
    def response(self) -> str:
        """Return outward response text separately from action memory text."""
        return self.action.response

    @property
    def memory_to_store(self) -> str:
        """Return the parsed action-memory content."""
        return self.action.memory_to_store


@dataclass(frozen=True)
class PostStepReflectionInput:
    """Validated inputs for one agent's literal post-step reflection."""

    agent_id: str
    source_step_id: str
    memory_stream: MemoryStream
    logical_timestamp: int
    conversation_context: tuple[str, ...]
    final_turn_result: CognitiveResult
    event_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("agent_id", "source_step_id"):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), field_name),
            )
        if self.agent_id not in FIXED_AGENT_IDS:
            raise CognitiveInputError(f"unsupported agent_id: {self.agent_id!r}")
        if not isinstance(self.memory_stream, MemoryStream):
            raise CognitiveInputError("memory_stream must be a MemoryStream")
        if self.memory_stream.run_id is None:
            raise CognitiveInputError(
                "post-step reflection requires a run-owned MemoryStream"
            )
        if self.memory_stream.agent_id != self.agent_id:
            raise CognitiveInputError(
                "memory-stream ownership does not match the requested agent"
            )
        if (
            isinstance(self.logical_timestamp, bool)
            or not isinstance(self.logical_timestamp, int)
            or self.logical_timestamp < 1
        ):
            raise CognitiveInputError(
                "post-step logical_timestamp must be a positive integer"
            )
        object.__setattr__(
            self,
            "conversation_context",
            _normalize_context(self.conversation_context),
        )
        if self.event_id is not None:
            object.__setattr__(
                self,
                "event_id",
                _require_text(self.event_id, "event_id"),
            )
        if not isinstance(self.final_turn_result, CognitiveResult):
            raise CognitiveInputError(
                "final_turn_result must be a CognitiveResult"
            )
        final_result = self.final_turn_result
        if (
            final_result.agent_id != self.agent_id
            or final_result.source_step_id != self.source_step_id
            or final_result.event_id != self.event_id
        ):
            raise CognitiveInputError(
                "final_turn_result identity does not match post-step reflection input"
            )
        if self.logical_timestamp < final_result.logical_timestamp:
            raise CognitiveInputError(
                "post-step logical_timestamp must not precede the final agent turn"
            )

        result_memories = (
            final_result.retrieval_result.selected_memories
            + final_result.created_memories
        )
        if any(
            memory.run_id != self.memory_stream.run_id
            for memory in result_memories
        ):
            raise CognitiveInputError(
                "final_turn_result does not belong to the MemoryStream run"
            )


@dataclass(frozen=True)
class PostStepReflectionResult:
    """Immutable output of one reflection-only post-step operation."""

    final_turn_id: str
    agent_id: str
    source_step_id: str
    source_step_type: str
    event_id: str | None
    retrieval_config_id: str
    run_id: str
    logical_timestamp: int
    conversation_context: tuple[str, ...]
    selected_memory_ids: tuple[str, ...]
    selected_memory_texts: tuple[str, ...]
    reflection: ReflectionOutput
    reflection_importance: ImportanceOutput
    reflection_memory: MemoryEntry

    def __post_init__(self) -> None:
        for field_name in (
            "final_turn_id",
            "agent_id",
            "source_step_id",
            "retrieval_config_id",
            "run_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise CognitiveResultValidationError(
                    f"{field_name} must be a non-empty string"
                )
        if self.agent_id not in FIXED_AGENT_IDS:
            raise CognitiveResultValidationError(
                f"unsupported result agent_id: {self.agent_id!r}"
            )
        if self.source_step_type not in STEP_TYPES:
            raise CognitiveResultValidationError(
                f"unsupported result source_step_type: {self.source_step_type!r}"
            )
        if self.source_step_type == "target_event":
            if not isinstance(self.event_id, str) or not self.event_id.strip():
                raise CognitiveResultValidationError(
                    "target-event reflection result requires event_id"
                )
        elif self.event_id is not None:
            raise CognitiveResultValidationError(
                "natural-interaction reflection result must not include event_id"
            )
        if (
            isinstance(self.logical_timestamp, bool)
            or not isinstance(self.logical_timestamp, int)
            or self.logical_timestamp < 1
        ):
            raise CognitiveResultValidationError(
                "post-step logical_timestamp must be a positive integer"
            )
        if not isinstance(self.conversation_context, tuple) or any(
            not isinstance(item, str) or not item.strip()
            for item in self.conversation_context
        ):
            raise CognitiveResultValidationError(
                "conversation_context must be a tuple of non-empty strings"
            )
        if not isinstance(self.selected_memory_ids, tuple) or any(
            not isinstance(memory_id, str) or not memory_id.strip()
            for memory_id in self.selected_memory_ids
        ):
            raise CognitiveResultValidationError(
                "selected_memory_ids must be a tuple of non-empty strings"
            )
        if not isinstance(self.selected_memory_texts, tuple) or any(
            not isinstance(text, str) or not text.strip()
            for text in self.selected_memory_texts
        ):
            raise CognitiveResultValidationError(
                "selected_memory_texts must be a tuple of non-empty strings"
            )
        if len(self.selected_memory_ids) != len(self.selected_memory_texts):
            raise CognitiveResultValidationError(
                "selected memory IDs and texts must have equal lengths"
            )
        if not isinstance(self.reflection, ReflectionOutput):
            raise CognitiveResultValidationError(
                "reflection must be a ReflectionOutput"
            )
        if not isinstance(self.reflection_importance, ImportanceOutput):
            raise CognitiveResultValidationError(
                "reflection_importance must be an ImportanceOutput"
            )
        if not isinstance(self.reflection_memory, MemoryEntry):
            raise CognitiveResultValidationError(
                "reflection_memory must be a MemoryEntry"
            )
        memory = self.reflection_memory
        if (
            memory.agent_id != self.agent_id
            or memory.run_id != self.run_id
            or memory.memory_type != "reflection"
            or memory.source_step_id != self.source_step_id
            or memory.source_step_type != self.source_step_type
            or memory.created_at != self.logical_timestamp
            or memory.last_accessed_at != self.logical_timestamp
            or memory.embedding is None
            or memory.text != self.reflection.reflection
            or memory.category != self.reflection.category
            or memory.importance != self.reflection_importance.importance
        ):
            raise CognitiveResultValidationError(
                "reflection memory does not match post-step reflection output"
            )

    @property
    def created_memories(self) -> tuple[MemoryEntry, ...]:
        """Expose the single reflection memory as an immutable tuple."""
        return (self.reflection_memory,)

    @property
    def created_memory(self) -> MemoryEntry:
        """Return the single created memory for controller persistence."""
        return self.reflection_memory


class CognitivePipeline:
    """Run one deterministic turn and commit memory only after full preflight."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        llm_client: LLMClient,
        embedding_provider: TextEmbeddingProvider,
        retrieval_engine: Any | None = None,
        category_vocabulary: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        if not isinstance(config, ExperimentConfig):
            raise CognitiveInputError("config must be an ExperimentConfig")
        validate_experiment_config(config)
        if not callable(getattr(llm_client, "generate", None)):
            raise CognitiveInputError("llm_client must provide generate(request)")
        if not callable(getattr(embedding_provider, "embed", None)):
            raise CognitiveInputError("embedding_provider must provide embed(text)")
        engine = (
            RetrievalEngine(config.settings)
            if retrieval_engine is None
            else retrieval_engine
        )
        if not callable(getattr(engine, "retrieve", None)):
            raise CognitiveInputError("retrieval_engine must provide retrieve(...)")

        self._config = config
        self._llm_client = llm_client
        self._embedding_provider = embedding_provider
        self._retrieval_engine = engine
        self._agents = {agent.agent_id: agent for agent in config.agents}
        self._category_vocabulary: dict[str, tuple[str, ...]] = {}
        if category_vocabulary is not None:
            if not isinstance(category_vocabulary, Mapping):
                raise CognitiveInputError("category_vocabulary must be a mapping")
            for agent_id, labels in category_vocabulary.items():
                if agent_id not in self._agents:
                    raise CognitiveInputError(
                        f"category_vocabulary references unknown agent_id: {agent_id!r}"
                    )
                normalized = tuple(dict.fromkeys(labels))
                if not normalized or not all(
                    isinstance(label, str) and label.strip() for label in normalized
                ):
                    raise CognitiveInputError(
                        f"category_vocabulary for {agent_id!r} must contain non-empty strings"
                    )
                self._category_vocabulary[agent_id] = normalized
        self._steps = {step.step_id: step for step in config.steps}
        self._events = {event.event_id: event for event in config.target_events}
        self._retrieval_configs = {
            item.config_id: item for item in config.retrieval_configs
        }
        self._planner = Planner(llm_client)
        self._actor = Actor(llm_client)
        self._reflector = Reflector(llm_client)

    @property
    def retrieval_engine(self) -> Any:
        """Expose the injected engine for focused orchestration inspection."""
        return self._retrieval_engine

    def run_turn(self, turn_input: CognitiveTurnInput) -> CognitiveResult:
        """Execute and validate every stage before committing stream mutations."""
        if not isinstance(turn_input, CognitiveTurnInput):
            raise CognitiveInputError("turn_input must be a CognitiveTurnInput")
        agent, step, event, retrieval_config = self._resolve_input(turn_input)
        memory_stream = turn_input.memory_stream
        initial_memories = memory_stream.list_memories()
        if any(
            turn_input.logical_timestamp < memory.last_accessed_at
            for memory in initial_memories
        ):
            raise CognitiveInputError(
                "logical_timestamp must not precede memory access timestamps"
            )

        perception = build_perception(
            agent=agent,
            step=step,
            logical_timestamp=turn_input.logical_timestamp,
            conversation_context=turn_input.conversation_context,
            event=event,
        )
        retrieval_query = build_retrieval_query(perception)
        retrieval_result = self._retrieval_engine.retrieve(
            memory_stream=memory_stream,
            query=retrieval_query.text,
            query_embedding=turn_input.query_embedding,
            retrieval_config=retrieval_config,
            current_time=turn_input.logical_timestamp,
        )
        if memory_stream.list_memories() != initial_memories:
            raise CognitiveStateError(
                "retrieval_engine mutated the memory stream before the commit boundary"
            )
        self._validate_retrieval_result(
            retrieval_result,
            initial_memories=initial_memories,
            query=retrieval_query,
            agent=agent,
            retrieval_config=retrieval_config,
        )

        llm_context = self._build_llm_context(
            perception,
            retrieval_result.selected_memories,
        )
        plan = self._planner.generate(llm_context)
        response_limit = self._response_limit(step)
        action = self._actor.generate(
            context=llm_context,
            plan=plan,
            response_limit=response_limit,
            previous_same_agent_response=turn_input.previous_same_agent_response,
        )
        reflection = (
            self._reflector.generate(
                context=llm_context,
                plan=plan,
                action=action,
                sentence_limit=self._config.settings.response_limits.reflection_sentences,
            )
            if turn_input.reflection_requested
            else None
        )

        action_importance: ImportanceOutput | None = None
        reflection_importance: ImportanceOutput | None = None
        created_memories: tuple[MemoryEntry, ...] = ()
        if step.memory_generation_enabled:
            action_importance = self._generate_importance(
                context=llm_context,
                text=action.memory_to_store,
                memory_kind="action",
            )
            action_embedding = self._generate_embedding(
                action.memory_to_store,
                expected_dimensions=len(turn_input.query_embedding),
            )
            reflection_embedding: tuple[float, ...] | None = None
            if reflection is not None:
                reflection_importance = self._generate_importance(
                    context=llm_context,
                    text=reflection.reflection,
                    memory_kind="reflection",
                )
                reflection_embedding = self._generate_embedding(
                    reflection.reflection,
                    expected_dimensions=len(turn_input.query_embedding),
                )
            created_memories = self._build_memories(
                memory_stream=memory_stream,
                initial_memories=initial_memories,
                turn_input=turn_input,
                step=step,
                action=action,
                action_importance=action_importance,
                action_embedding=action_embedding,
                reflection=reflection,
                reflection_importance=reflection_importance,
                reflection_embedding=reflection_embedding,
            )

        result = CognitiveResult(
            turn_id=turn_input.turn_id,
            agent_id=agent.agent_id,
            source_step_id=step.step_id,
            source_step_type=step.step_type,
            event_id=event.event_id if event is not None else None,
            retrieval_config_id=retrieval_config.config_id,
            logical_timestamp=turn_input.logical_timestamp,
            perception=perception,
            retrieval_query=retrieval_query,
            retrieval_result=retrieval_result,
            selected_memory_ids=retrieval_result.selected_memory_ids,
            plan=plan,
            action=action,
            reflection=reflection,
            reflection_requested=turn_input.reflection_requested,
            memory_generation_enabled=step.memory_generation_enabled,
            action_importance=action_importance,
            reflection_importance=reflection_importance,
            created_memories=created_memories,
            accessed_memory_ids=retrieval_result.selected_memory_ids,
        )

        self._preflight_mutations(
            memory_stream=memory_stream,
            initial_memories=initial_memories,
            accessed_memory_ids=result.accessed_memory_ids,
            logical_timestamp=turn_input.logical_timestamp,
            created_memories=result.created_memories,
        )
        if memory_stream.list_memories() != initial_memories:
            raise CognitiveStateError(
                "memory stream changed before the approved commit boundary"
            )

        # Nothing that can invoke a cognitive dependency occurs after this point.
        memory_stream.update_last_accessed_at(
            result.accessed_memory_ids,
            accessed_at=turn_input.logical_timestamp,
        )
        for memory in result.created_memories:
            memory_stream.add_memory(memory)
        return result

    def run_post_step_reflection(
        self,
        reflection_input: PostStepReflectionInput,
    ) -> PostStepReflectionResult:
        """Reflect once after a completed step without action or retrieval work."""
        if not isinstance(reflection_input, PostStepReflectionInput):
            raise CognitiveInputError(
                "reflection_input must be a PostStepReflectionInput"
            )
        agent, step, event = self._resolve_post_step_reflection_input(
            reflection_input
        )
        final_result = reflection_input.final_turn_result
        memory_stream = reflection_input.memory_stream
        initial_memories = memory_stream.list_memories()
        if any(
            reflection_input.logical_timestamp < memory.last_accessed_at
            for memory in initial_memories
        ):
            raise CognitiveInputError(
                "logical_timestamp must not precede memory access timestamps"
            )
        if any(
            memory.memory_type == "reflection"
            and memory.source_step_id == step.step_id
            for memory in initial_memories
        ):
            raise CognitiveInputError(
                "post-step reflection already exists for this agent and step"
            )

        self._validate_complete_step_context(
            reflection_input=reflection_input,
            agent=agent,
            step=step,
        )
        selected_memories = final_result.retrieval_result.selected_memories
        perception = build_perception(
            agent=agent,
            step=step,
            logical_timestamp=reflection_input.logical_timestamp,
            conversation_context=reflection_input.conversation_context,
            event=event,
        )
        llm_context = self._build_llm_context(perception, selected_memories)
        reflection = self._reflector.generate(
            context=llm_context,
            plan=final_result.plan,
            action=final_result.action,
            sentence_limit=self._config.settings.response_limits.reflection_sentences,
        )
        reflection_importance = self._generate_importance(
            context=llm_context,
            text=reflection.reflection,
            memory_kind="reflection",
        )
        reflection_embedding = self._generate_embedding(
            reflection.reflection,
            expected_dimensions=self._post_step_embedding_dimensions(final_result),
        )
        reflection_memory = self._build_post_step_reflection_memory(
            memory_stream=memory_stream,
            initial_memories=initial_memories,
            reflection_input=reflection_input,
            step=step,
            reflection=reflection,
            reflection_importance=reflection_importance,
            reflection_embedding=reflection_embedding,
        )
        result = PostStepReflectionResult(
            final_turn_id=final_result.turn_id,
            agent_id=agent.agent_id,
            source_step_id=step.step_id,
            source_step_type=step.step_type,
            event_id=event.event_id if event is not None else None,
            retrieval_config_id=final_result.retrieval_config_id,
            run_id=memory_stream.run_id,
            logical_timestamp=reflection_input.logical_timestamp,
            conversation_context=reflection_input.conversation_context,
            selected_memory_ids=final_result.selected_memory_ids,
            selected_memory_texts=tuple(
                memory.text for memory in selected_memories
            ),
            reflection=reflection,
            reflection_importance=reflection_importance,
            reflection_memory=reflection_memory,
        )

        if memory_stream.list_memories() != initial_memories:
            raise CognitiveStateError(
                "memory stream changed before post-step reflection preflight"
            )
        self._preflight_mutations(
            memory_stream=memory_stream,
            initial_memories=initial_memories,
            accessed_memory_ids=(),
            logical_timestamp=reflection_input.logical_timestamp,
            created_memories=result.created_memories,
        )
        if memory_stream.list_memories() != initial_memories:
            raise CognitiveStateError(
                "memory stream changed before the approved commit boundary"
            )

        # All fallible cognitive and validation work is complete before commit.
        memory_stream.add_memory(result.reflection_memory)
        return result

    run = run_turn

    def _resolve_post_step_reflection_input(
        self,
        reflection_input: PostStepReflectionInput,
    ) -> tuple[AgentProfile, SimulationStep, TargetEvent | None]:
        try:
            agent = self._agents[reflection_input.agent_id]
        except KeyError as exc:
            raise CognitiveInputError(
                f"unknown agent_id: {reflection_input.agent_id!r}"
            ) from exc
        try:
            step = self._steps[reflection_input.source_step_id]
        except KeyError as exc:
            raise CognitiveInputError(
                f"unknown source_step_id: {reflection_input.source_step_id!r}"
            ) from exc

        final_result = reflection_input.final_turn_result
        if (
            final_result.source_step_type != step.step_type
            or final_result.agent_id != agent.agent_id
            or final_result.source_step_id != step.step_id
        ):
            raise CognitiveInputError(
                "final_turn_result does not match the configured agent and step"
            )
        if final_result.retrieval_config_id not in self._retrieval_configs:
            raise CognitiveInputError(
                "final_turn_result references an unknown retrieval configuration"
            )
        if agent.agent_id not in step.participants:
            raise CognitiveInputError(
                f"agent {agent.agent_id!r} is not a participant in {step.step_id!r}"
            )
        if not step.memory_generation_enabled:
            raise CognitiveInputError(
                "post-step reflection requires enabled memory generation"
            )
        if final_result.reflection_requested or final_result.reflection is not None:
            raise CognitiveInputError(
                "post-step reflection requires an ordinary main turn without inline reflection"
            )
        if (
            len(final_result.created_memories) != 1
            or final_result.created_memories[0].memory_type != "action"
        ):
            raise CognitiveInputError(
                "final_turn_result must contain exactly one action memory"
            )

        current_memories = reflection_input.memory_stream.list_memories()
        final_action_memory = final_result.created_memories[0]
        if not current_memories or current_memories[-1] != final_action_memory:
            raise CognitiveInputError(
                "final_turn_result is not the agent's final committed main turn"
            )

        event: TargetEvent | None = None
        if isinstance(step, TargetEventStep):
            try:
                event = self._events[step.event_id]
            except KeyError as exc:
                raise CognitiveInputError(
                    f"unknown event_id referenced by step: {step.event_id!r}"
                ) from exc
            if reflection_input.event_id != event.event_id:
                raise CognitiveInputError(
                    "post-step event_id does not match the configured step event"
                )
        elif reflection_input.event_id is not None:
            raise CognitiveInputError(
                "natural-interaction reflection must not include an event_id"
            )
        return agent, step, event

    @staticmethod
    def _validate_complete_step_context(
        *,
        reflection_input: PostStepReflectionInput,
        agent: AgentProfile,
        step: SimulationStep,
    ) -> None:
        context = reflection_input.conversation_context
        expected_count = (
            step.total_turns
            if isinstance(step, NaturalInteractionBlockStep)
            else len(step.participants)
        )
        if len(context) != expected_count:
            raise CognitiveInputError(
                "post-step conversation_context must contain every main response"
            )

        final_result = reflection_input.final_turn_result
        prior_context = final_result.perception.conversation_context
        if context[: len(prior_context)] != prior_context:
            raise CognitiveInputError(
                "post-step conversation_context does not preserve final-turn history"
            )
        if len(prior_context) >= len(context):
            raise CognitiveInputError(
                "post-step conversation_context is missing the final agent response"
            )
        expected_response = f"{agent.display_name}: {final_result.response}"
        if context[len(prior_context)] != expected_response:
            raise CognitiveInputError(
                "post-step conversation_context does not contain the speaker-labelled "
                "final agent response in turn order"
            )

    @staticmethod
    def _post_step_embedding_dimensions(final_result: CognitiveResult) -> int:
        memories = (
            final_result.retrieval_result.selected_memories
            + final_result.created_memories
        )
        dimensions = {
            len(memory.embedding)
            for memory in memories
            if memory.embedding is not None
        }
        if not dimensions:
            raise CognitiveResultValidationError(
                "final_turn_result does not establish embedding dimensions"
            )
        if len(dimensions) != 1:
            raise CognitiveResultValidationError(
                "final_turn_result contains inconsistent embedding dimensions"
            )
        return dimensions.pop()

    def _build_post_step_reflection_memory(
        self,
        *,
        memory_stream: MemoryStream,
        initial_memories: tuple[MemoryEntry, ...],
        reflection_input: PostStepReflectionInput,
        step: SimulationStep,
        reflection: ReflectionOutput,
        reflection_importance: ImportanceOutput,
        reflection_embedding: tuple[float, ...],
    ) -> MemoryEntry:
        factory = MemoryFactory(
            reflection_input.agent_id,
            run_id=memory_stream.run_id,
            start_sequence=self._next_memory_sequence(
                reflection_input.agent_id,
                initial_memories,
            ),
        )
        return factory.create_memory(
            memory_type="reflection",
            text=reflection.reflection,
            category=reflection.category,
            importance=reflection_importance.importance,
            created_at=reflection_input.logical_timestamp,
            source_step_id=step.step_id,
            source_step_type=step.step_type,
            embedding=reflection_embedding,
        )

    def _resolve_input(
        self,
        turn_input: CognitiveTurnInput,
    ) -> tuple[AgentProfile, SimulationStep, TargetEvent | None, RetrievalConfig]:
        try:
            agent = self._agents[turn_input.agent_id]
        except KeyError as exc:
            raise CognitiveInputError(
                f"unknown agent_id: {turn_input.agent_id!r}"
            ) from exc
        try:
            step = self._steps[turn_input.source_step_id]
        except KeyError as exc:
            raise CognitiveInputError(
                f"unknown source_step_id: {turn_input.source_step_id!r}"
            ) from exc
        try:
            retrieval_config = self._retrieval_configs[
                turn_input.retrieval_config_id
            ]
        except KeyError as exc:
            raise CognitiveInputError(
                f"unknown retrieval_config_id: {turn_input.retrieval_config_id!r}"
            ) from exc

        if turn_input.memory_stream.agent_id != agent.agent_id:
            raise CognitiveInputError(
                "memory-stream ownership does not match the requested agent"
            )
        if agent.agent_id not in step.participants:
            raise CognitiveInputError(
                f"agent {agent.agent_id!r} is not a participant in {step.step_id!r}"
            )

        event: TargetEvent | None = None
        if isinstance(step, TargetEventStep):
            try:
                event = self._events[step.event_id]
            except KeyError as exc:
                raise CognitiveInputError(
                    f"unknown event_id referenced by step: {step.event_id!r}"
                ) from exc
            if turn_input.event_id is not None and turn_input.event_id != event.event_id:
                raise CognitiveInputError(
                    f"event_id {turn_input.event_id!r} does not match step event "
                    f"{event.event_id!r}"
                )
        elif turn_input.event_id is not None:
            raise CognitiveInputError(
                "natural-interaction turn must not include an event_id"
            )
        return agent, step, event, retrieval_config

    def _validate_retrieval_result(
        self,
        retrieval_result: object,
        *,
        initial_memories: tuple[MemoryEntry, ...],
        query: RetrievalQuery,
        agent: AgentProfile,
        retrieval_config: RetrievalConfig,
    ) -> None:
        if not isinstance(retrieval_result, RetrievalResult):
            raise CognitiveResultValidationError(
                "retrieval engine must return a RetrievalResult"
            )
        if retrieval_result.query != query.text:
            raise CognitiveResultValidationError(
                "RetrievalResult query does not match the requested query"
            )
        if retrieval_result.agent_id != agent.agent_id:
            raise CognitiveResultValidationError(
                "RetrievalResult agent does not match the requested agent"
            )
        if retrieval_result.retrieval_config_id != retrieval_config.config_id:
            raise CognitiveResultValidationError(
                "RetrievalResult configuration does not match the requested configuration"
            )
        if retrieval_result.applied_weights != retrieval_config.weights:
            raise CognitiveResultValidationError(
                "RetrievalResult weights do not match the requested configuration"
            )
        if retrieval_result.top_k != self._config.settings.top_k:
            raise CognitiveResultValidationError(
                "RetrievalResult Top-K does not match validated settings"
            )
        if retrieval_result.candidate_count != len(initial_memories):
            raise CognitiveResultValidationError(
                "RetrievalResult candidate count does not match the memory stream"
            )

        initial_by_id = {memory.memory_id: memory for memory in initial_memories}
        result_ids = tuple(
            candidate.memory_id for candidate in retrieval_result.candidates
        )
        if set(result_ids) != set(initial_by_id):
            raise CognitiveResultValidationError(
                "RetrievalResult candidates must be exactly the input memory stream"
            )
        if any(
            candidate.memory != initial_by_id[candidate.memory_id]
            for candidate in retrieval_result.candidates
        ):
            raise CognitiveResultValidationError(
                "RetrievalResult contains memory data not present in the input stream"
            )

    def _build_llm_context(
        self,
        perception: Perception,
        selected_memories: tuple[MemoryEntry, ...],
    ) -> LLMContext:
        return LLMContext(
            agent_id=perception.agent_id,
            display_name=perception.display_name,
            role=perception.role,
            archetype=perception.archetype,
            persona_summary=perception.persona_summary,
            traits=perception.traits,
            goals=perception.goals,
            communication_style=perception.communication_style,
            motivations=perception.motivations,
            source_step_id=perception.source_step_id,
            source_step_type=perception.source_step_type,
            step_title=perception.step_title,
            step_description=perception.step_description,
            day=perception.day,
            period=perception.period,
            logical_timestamp=perception.logical_timestamp,
            observation_text=perception.observation_text,
            conversation_context=perception.conversation_context,
            selected_memory_texts=tuple(memory.text for memory in selected_memories),
            category_vocabulary=self._category_vocabulary.get(
                perception.agent_id, ()
            ),
            event_id=perception.event_id,
            event_title=perception.event_title,
            event_description=perception.event_description,
        )

    def _response_limit(self, step: SimulationStep) -> int:
        limits = self._config.settings.response_limits
        return (
            limits.target_event_max_paragraphs
            if step.step_type == "target_event"
            else limits.natural_interaction_sentences
        )

    def _generate_importance(
        self,
        *,
        context: LLMContext,
        text: str,
        memory_kind: str,
    ) -> ImportanceOutput:
        request = LLMRequest(
            call_type=LLMCallType.IMPORTANCE_SCORE,
            context=context,
            memory_text=text,
            memory_kind=memory_kind,
        )
        return parse_importance_output(self._llm_client.generate(request))

    def _generate_embedding(
        self,
        text: str,
        *,
        expected_dimensions: int,
    ) -> tuple[float, ...]:
        embedding = validate_vector(
            self._embedding_provider.embed(text),
            field_name="generated_memory_embedding",
        )
        if not any(value != 0.0 for value in embedding):
            raise RetrievalScoringError("generated_memory_embedding must be non-zero")
        if len(embedding) != expected_dimensions:
            raise RetrievalScoringError(
                "generated memory embedding dimensions must match query_embedding"
            )
        return embedding

    @staticmethod
    def _next_memory_sequence(
        agent_id: str,
        initial_memories: tuple[MemoryEntry, ...],
    ) -> int:
        pattern = re.compile(rf"^mem_{re.escape(agent_id)}_(\d+)$")
        existing_sequences = [
            int(match.group(1))
            for memory in initial_memories
            if (match := pattern.fullmatch(memory.memory_id)) is not None
        ]
        return max((len(initial_memories), *existing_sequences), default=0) + 1

    def _build_memories(
        self,
        *,
        memory_stream: MemoryStream,
        initial_memories: tuple[MemoryEntry, ...],
        turn_input: CognitiveTurnInput,
        step: SimulationStep,
        action: ActionOutput,
        action_importance: ImportanceOutput,
        action_embedding: tuple[float, ...],
        reflection: ReflectionOutput | None,
        reflection_importance: ImportanceOutput | None,
        reflection_embedding: tuple[float, ...] | None,
    ) -> tuple[MemoryEntry, ...]:
        factory = MemoryFactory(
            turn_input.agent_id,
            run_id=memory_stream.run_id,
            start_sequence=self._next_memory_sequence(
                turn_input.agent_id,
                initial_memories,
            ),
        )
        memories = [
            factory.create_memory(
                memory_type="action",
                text=action.memory_to_store,
                category=action.category,
                importance=action_importance.importance,
                created_at=turn_input.logical_timestamp,
                source_step_id=step.step_id,
                source_step_type=step.step_type,
                embedding=action_embedding,
            )
        ]
        if reflection is not None:
            if reflection_importance is None or reflection_embedding is None:
                raise CognitiveResultValidationError(
                    "reflection enrichment must be complete before memory creation"
                )
            memories.append(
                factory.create_memory(
                    memory_type="reflection",
                    text=reflection.reflection,
                    category=reflection.category,
                    importance=reflection_importance.importance,
                    created_at=turn_input.logical_timestamp,
                    source_step_id=step.step_id,
                    source_step_type=step.step_type,
                    embedding=reflection_embedding,
                )
            )
        return tuple(memories)

    @staticmethod
    def _preflight_mutations(
        *,
        memory_stream: MemoryStream,
        initial_memories: tuple[MemoryEntry, ...],
        accessed_memory_ids: tuple[str, ...],
        logical_timestamp: int,
        created_memories: tuple[MemoryEntry, ...],
    ) -> None:
        shadow = MemoryStream(memory_stream.agent_id, run_id=memory_stream.run_id)
        for memory in initial_memories:
            shadow.add_memory(memory)
        shadow.update_last_accessed_at(
            accessed_memory_ids,
            accessed_at=logical_timestamp,
        )
        for memory in created_memories:
            shadow.add_memory(memory)
