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

Phase 01 provides the repository structure and a placeholder command-line
interface only. Simulation, memory, retrieval, LLM, evaluation, storage, and
visualizer behavior will be implemented in later phases.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Commands

Run the Phase 01 placeholder:

```bash
python main.py
python main.py --help
python main.py validate-config
```

The `validate-config` command is intentionally a placeholder in this phase.

Run the tests:

```bash
python -m pytest
```

## Documentation

- [Architecture contract](docs/ARCHITECTURE_CONTRACT.md)
- [Final clean architecture blueprint](docs/Document_01_Final_Clean_Architecture_Blueprint.md)
- [Final clean implementation specification](docs/Document_02_Final_Clean_Implementation_Specification.md)
