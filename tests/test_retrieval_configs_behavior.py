import pytest

from src.config_loader import load_experiment_config
from src.memory import MemoryFactory, MemoryStream
from src.retrieval import RECENCY_DECAY_FACTOR, RetrievalEngine


def build_component_dominance_stream() -> MemoryStream:
    factory = MemoryFactory("active_student", run_id="behavior_run")
    stream = MemoryStream("active_student", run_id="behavior_run")
    definitions = (
        (
            "recency",
            1,
            10,
            (0.0, -1.0),
        ),
        (
            "importance",
            10,
            0,
            (0.0, -1.0),
        ),
        (
            "relevance",
            1,
            0,
            (0.0, 1.0),
        ),
    )
    for label, importance, last_accessed_at, embedding in definitions:
        stream.add_memory(
            factory.create_memory(
                memory_type="observation",
                text=f"{label} dominant memory",
                category="social_learning",
                importance=importance,
                created_at=0,
                last_accessed_at=last_accessed_at,
                source_step_id="day_01_natural_01",
                source_step_type="natural_interaction_block",
                embedding=embedding,
            )
        )
    return stream


def retrieve_all_configurations():
    experiment = load_experiment_config()
    engine = RetrievalEngine(experiment.settings)
    stream = build_component_dominance_stream()
    return experiment, {
        config.config_id: engine.retrieve(
            memory_stream=stream,
            query="Which classroom memory should guide the response?",
            query_embedding=(0.0, 1.0),
            retrieval_config=config,
            current_time=10,
        )
        for config in experiment.retrieval_configs
    }


def test_loaded_retrieval_configurations_and_top_k_remain_fixed() -> None:
    experiment = load_experiment_config()

    assert experiment.settings.top_k == 5
    assert {
        config.config_id: (
            config.weights.recency,
            config.weights.importance,
            config.weights.relevance,
        )
        for config in experiment.retrieval_configs
    } == {
        "balanced": (0.33, 0.33, 0.33),
        "recency_heavy": (0.60, 0.20, 0.20),
        "importance_heavy": (0.20, 0.60, 0.20),
        "relevance_heavy": (0.20, 0.20, 0.60),
    }


def test_same_recency_formula_is_used_across_all_four_configurations() -> None:
    _, results = retrieve_all_configurations()
    recencies_by_config = {
        config_id: {
            candidate.memory.text: candidate.raw_recency
            for candidate in result.candidates
        }
        for config_id, result in results.items()
    }

    assert len(set(tuple(sorted(values.items())) for values in recencies_by_config.values())) == 1
    common = next(iter(recencies_by_config.values()))
    assert common["recency dominant memory"] == 1.0
    assert common["importance dominant memory"] == pytest.approx(
        RECENCY_DECAY_FACTOR**10
    )


@pytest.mark.parametrize(
    ("config_id", "expected_winner"),
    [
        ("recency_heavy", "recency dominant memory"),
        ("importance_heavy", "importance dominant memory"),
        ("relevance_heavy", "relevance dominant memory"),
    ],
)
def test_heavy_configuration_favors_its_dominant_component(
    config_id: str,
    expected_winner: str,
) -> None:
    _, results = retrieve_all_configurations()

    assert results[config_id].candidates[0].memory.text == expected_winner


def test_only_applied_weights_change_component_scores_across_configurations() -> None:
    _, results = retrieve_all_configurations()

    component_snapshots = []
    for result in results.values():
        by_memory_id = {
            candidate.memory_id: (
                candidate.raw_recency,
                candidate.raw_importance,
                candidate.raw_relevance,
                candidate.normalized_recency,
                candidate.normalized_importance,
                candidate.normalized_relevance,
            )
            for candidate in result.candidates
        }
        component_snapshots.append(by_memory_id)

        for candidate in result.candidates:
            expected = (
                result.applied_weights.recency * candidate.normalized_recency
                + result.applied_weights.importance
                * candidate.normalized_importance
                + result.applied_weights.relevance
                * candidate.normalized_relevance
            )
            assert candidate.final_score == pytest.approx(expected)

    assert all(snapshot == component_snapshots[0] for snapshot in component_snapshots)


def test_repeated_configuration_retrieval_is_deterministic() -> None:
    _, first = retrieve_all_configurations()
    _, second = retrieve_all_configurations()

    assert first == second
