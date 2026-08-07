"""Shared deterministic Phase 08 integration fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.llm import DummyEmbeddingProvider
from src.simulation import RunController, build_deterministic_dummy_llm_client


@pytest.fixture(scope="session")
def phase08_pilot_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
):
    """Execute one shared offline Day 1-2 pilot for read-only assertions."""
    output_root = tmp_path_factory.mktemp("phase08_pilot")
    llm_client = build_deterministic_dummy_llm_client()
    embedding_provider = DummyEmbeddingProvider()
    result = RunController(
        llm_client=llm_client,
        embedding_provider=embedding_provider,
    ).run(
        output_root=output_root,
        experiment_id="phase08_tests",
        run_id="balanced_rep01_pilot",
        retrieval_config_id="balanced",
        repetition_index=1,
        days=(1, 2),
    )
    return result, llm_client, embedding_provider


@pytest.fixture(scope="session")
def phase08_full_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
):
    """Execute one shared complete offline run for fixed-total assertions."""
    output_root: Path = tmp_path_factory.mktemp("phase08_full")
    llm_client = build_deterministic_dummy_llm_client()
    embedding_provider = DummyEmbeddingProvider()
    result = RunController(
        llm_client=llm_client,
        embedding_provider=embedding_provider,
    ).run(
        output_root=output_root,
        experiment_id="phase08_tests",
        run_id="balanced_rep01_full",
        retrieval_config_id="balanced",
        repetition_index=1,
    )
    return result, llm_client, embedding_provider
