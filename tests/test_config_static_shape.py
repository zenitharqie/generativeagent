import json
from pathlib import Path


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "config"
AGENT_IDS = {
    "teacher",
    "active_student",
    "quiet_student",
    "academic_student",
}
EVENT_IDS = ["event_01", "event_02", "event_03", "event_04", "event_05"]
WEIGHT_KEYS = {"recency", "importance", "relevance"}
MEMORY_CATEGORIES = {
    "teacher_guidance",
    "classroom_norms",
    "social_learning",
    "quiet_style",
    "academic_focus",
    "exam_concern",
    "study_strategy",
    "question_asking",
}


def load_config(filename: str) -> dict:
    with (CONFIG_DIRECTORY / filename).open(encoding="utf-8") as config_file:
        return json.load(config_file)


def test_agents_have_fixed_ids_and_required_fields() -> None:
    agents = load_config("agents.json")["agents"]
    required_fields = {
        "agent_id",
        "display_name",
        "role",
        "archetype",
        "persona_summary",
        "traits",
        "goals",
        "communication_style",
        "motivations",
        "seed_memories",
    }

    assert len(agents) == 4
    assert {agent["agent_id"] for agent in agents} == AGENT_IDS
    assert all(required_fields <= agent.keys() for agent in agents)
    assert all(agent["seed_memories"] for agent in agents)


def test_agent_names_and_seed_memory_categories_are_stable() -> None:
    agents = load_config("agents.json")["agents"]
    expected_names = {
        "teacher": "Ms. Sunday",
        "active_student": "Monday",
        "quiet_student": "Tuesday",
        "academic_student": "Wednesday",
    }

    assert {agent["agent_id"]: agent["display_name"] for agent in agents} == expected_names
    assert all(
        memory["category"] in MEMORY_CATEGORIES
        for agent in agents
        for memory in agent["seed_memories"]
    )


def test_timeline_has_required_step_counts_and_natural_block_controls() -> None:
    steps = load_config("simulation_steps.json")["steps"]
    natural_steps = [
        step for step in steps if step["step_type"] == "natural_interaction_block"
    ]
    target_steps = [step for step in steps if step["step_type"] == "target_event"]

    assert len(steps) == 12
    assert len(natural_steps) == 7
    assert len(target_steps) == 5
    assert all(step["turns_per_agent"] == 3 for step in natural_steps)
    assert all(step["total_turns"] == 12 for step in natural_steps)
    assert all(step["teacher_opens_block"] is True for step in natural_steps)
    assert all(
        step["turn_policy"] in {"controlled_fixed", "controlled_semi_dynamic"}
        for step in natural_steps
    )
    assert all(step["evaluation_enabled"] is False for step in natural_steps)
    assert all(step["memory_generation_enabled"] is True for step in natural_steps)
    assert all(set(step["participants"]) == AGENT_IDS for step in steps)


def test_timeline_has_required_daily_shape_and_order() -> None:
    steps = load_config("simulation_steps.json")["steps"]

    day_1_types = [step["step_type"] for step in steps if step["day"] == 1]
    assert day_1_types == ["natural_interaction_block", "natural_interaction_block"]

    for day in range(2, 7):
        day_types = [step["step_type"] for step in steps if step["day"] == day]
        assert day_types == ["natural_interaction_block", "target_event"]


def test_target_steps_reference_all_events_in_fixed_order() -> None:
    steps = load_config("simulation_steps.json")["steps"]
    target_steps = [step for step in steps if step["step_type"] == "target_event"]
    events = load_config("target_events.json")["events"]

    assert [step["event_id"] for step in target_steps] == EVENT_IDS
    assert [event["event_id"] for event in events] == EVENT_IDS
    assert all(step["evaluation_enabled"] is True for step in target_steps)
    assert all(step["memory_generation_enabled"] is True for step in target_steps)


def test_target_event_definitions_are_complete_and_match_timeline() -> None:
    steps = load_config("simulation_steps.json")["steps"]
    target_steps = [step for step in steps if step["step_type"] == "target_event"]
    events = load_config("target_events.json")["events"]
    required_fields = {
        "event_id",
        "title",
        "description",
        "expected_memory_themes",
        "evaluation_role",
    }

    assert all(required_fields <= event.keys() for event in events)
    assert all(event["expected_memory_themes"] for event in events)
    assert all(set(event["expected_memory_themes"]) <= MEMORY_CATEGORIES for event in events)
    assert [
        (step["event_id"], step["title"], step["description"])
        for step in target_steps
    ] == [
        (event["event_id"], event["title"], event["description"])
        for event in events
    ]


def test_retrieval_configurations_use_only_fixed_weights() -> None:
    configurations = load_config("retrieval_configs.json")["configurations"]
    expected = {
        "balanced": {"recency": 0.33, "importance": 0.33, "relevance": 0.33},
        "recency_heavy": {"recency": 0.60, "importance": 0.20, "relevance": 0.20},
        "importance_heavy": {"recency": 0.20, "importance": 0.60, "relevance": 0.20},
        "relevance_heavy": {"recency": 0.20, "importance": 0.20, "relevance": 0.60},
    }

    assert len(configurations) == 4
    assert {item["config_id"]: item["weights"] for item in configurations} == expected
    assert all(set(item["weights"]) == WEIGHT_KEYS for item in configurations)


def test_settings_lock_architecture_constants_and_evaluation_scope() -> None:
    settings = load_config("settings.json")

    assert settings["top_k"] == 5
    assert settings["repetitions"] == 3
    assert settings["simulation_days"] == 6
    assert settings["natural_blocks_total"] == 7
    assert settings["target_events_total"] == 5
    assert settings["turns_per_agent_per_natural_block"] == 3
    assert settings["total_turns_per_natural_block"] == 12
    assert settings["teacher_opens_natural_block"] is True
    assert settings["natural_block_primary_evaluation"] is False
    assert settings["target_event_primary_evaluation"] is True
    assert settings["primary_mra_step_type"] == "target_event"
    assert settings["primary_bcr_step_type"] == "target_event"
    assert settings["dummy_mode_default"] is True
    assert settings["allow_real_llm_default"] is False
    assert settings["allow_full_experiment_default"] is False
    assert settings["response_limits"] == {
        "natural_interaction_sentences": 2,
        "target_event_max_paragraphs": 1,
        "reflection_sentences": 1,
    }


def test_ground_truth_is_target_only_and_covers_each_event_agent_pair() -> None:
    ground_truth = load_config("ground_truth.json")
    targets = ground_truth["targets"]
    pairs = {(target["event_id"], target["agent_id"]) for target in targets}
    expected_pairs = {(event_id, agent_id) for event_id in EVENT_IDS for agent_id in AGENT_IDS}

    assert ground_truth["evaluation_step_type"] == "target_event"
    assert ground_truth["status"] == "initial_configurable"
    assert ground_truth["freeze_required_before_final_experiment"] is True
    assert len(targets) == 20
    assert pairs == expected_pairs
    assert all(target["expected_categories"] for target in targets)
    assert all(set(target["expected_categories"]) <= MEMORY_CATEGORIES for target in targets)
