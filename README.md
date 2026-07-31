# Classroom-Based Generative Agents

This repository is the clean final thesis implementation for a controlled,
text-based classroom simulation used to study memory retrieval weighting in
LLM-based generative agents.

The project is an experimental testbed, not a full reproduction of Smallville
or Reverie. It intentionally excludes a spatial world, pathfinding, daily-life
scheduling, and a live simulation frontend.

## Research Boundary

- Natural interaction blocks provide controlled memory-building context.
- Target events are the primary evaluation anchors for Memory Retrieval
  Accuracy (MRA) and Behaviour Consistency Rate (BCR).
- Retrieval weight configuration is the only independent variable. All other
  experimental conditions remain controlled.
- The architecture contract in
  [`docs/ARCHITECTURE_CONTRACT.md`](docs/ARCHITECTURE_CONTRACT.md) is binding.

## Current Status

Phases 01–03 are complete and tested:

- **Phase 01 — Architecture Contract and Repository Skeleton:** provides the
  documented project boundary, package layout, initial command-line interface,
  and repository scaffold for later phases.
- **Phase 02 — Agent Initialization Configuration:** defines the four fixed
  agents, six-day hybrid timeline, five target events, four retrieval-weight
  configurations, fixed experiment settings, and initial target-event ground
  truth in JSON configuration files.
- **Phase 03 — Domain Models and Config Validation:** provides typed models for
  agents, personas, seed memories, simulation steps, target events, retrieval
  weights, experiment settings, and the future memory-entry structure. It also
  loads all configuration files and validates cross-file architecture
  invariants before any simulation or API call.

The next implementation phase is **Phase 04 — Memory Stream**. The current
repository has 60 passing tests, and `validate-config` accepts the committed
configuration.

## Implemented Configuration

The `config/` directory contains the controlled experiment definition:

- `agents.json`: exactly four agents (`teacher`, `active_student`,
  `quiet_student`, and `academic_student`) with stable personas and seed
  memories.
- `simulation_steps.json`: a 12-step, six-day schedule with seven natural
  interaction blocks and five target-event steps.
- `target_events.json`: the five fixed evaluation events in their required
  order.
- `retrieval_configs.json`: balanced, recency-heavy, importance-heavy, and
  relevance-heavy weight conditions.
- `settings.json`: fixed constants including Top-K 5, three repetitions, turn
  quotas, evaluation scope, response limits, and safe dummy-mode defaults.
- `ground_truth.json`: initial configurable category sets for target-event MRA;
  these values must be reviewed and frozen before the final experiment.

## Validation Scope

The `validate-config` command parses the JSON files into typed domain models and
rejects invalid experiment structures, including incorrect agent IDs, timeline
shape, fixed constants, natural-block controls, target-event references,
retrieval weights, evaluation scope, ground-truth coverage, and unsafe execution
defaults.

## Known Limitations

Phases 01–03 establish the architecture, configuration, typed models, and
validation layer only. The following runtime behavior is intentionally not yet
implemented:

- memory-stream insertion, ownership, and access operations;
- normalized retrieval scoring and Top-K selection;
- cognitive and LLM pipelines;
- natural-block, target-event, and full simulation execution;
- runtime logging, storage, MRA/BCR evaluation, experiment execution, analysis,
  and read-only replay visualization.

The `MemoryEntry` model and retrieval-weight models are schemas for later
phases; they do not yet perform memory-stream or retrieval operations.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Commands

Run the repository CLI or display its help:

```bash
python3 main.py
python3 main.py --help
```

Load and validate all Phase 02 configuration files against the Phase 03 domain
models and architecture invariants:

```bash
python3 main.py validate-config
```

To validate an alternate configuration directory:

```bash
python3 main.py validate-config --config-dir path/to/config
```

Run the tests:

```bash
python3 -m pytest
```

## Documentation

- [Architecture contract](docs/ARCHITECTURE_CONTRACT.md)
- [Final clean architecture blueprint](docs/Document_01_Final_Clean_Architecture_Blueprint.md)
- [Final clean implementation specification](docs/Document_02_Final_Clean_Implementation_Specification.md)
