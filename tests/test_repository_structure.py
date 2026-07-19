from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = (
    "config",
    "prompts",
    "src",
    "src/domain",
    "src/config_loader",
    "src/memory",
    "src/retrieval",
    "src/llm",
    "src/cognition",
    "src/simulation",
    "src/logging_system",
    "src/evaluation",
    "src/storage",
    "src/visualizer",
    "tests",
    "data",
    "logs",
    "resources",
    "visualizer",
    "docs",
)

REQUIRED_FILES = (
    "README.md",
    ".gitignore",
    ".env.example",
    "requirements.txt",
    "main.py",
    "docs/ARCHITECTURE_CONTRACT.md",
    "docs/Document_01_Final_Clean_Architecture_Blueprint.md",
    "docs/Document_02_Final_Clean_Implementation_Specification.md",
)

PACKAGE_DIRECTORIES = (
    "src",
    "src/domain",
    "src/config_loader",
    "src/memory",
    "src/retrieval",
    "src/llm",
    "src/cognition",
    "src/simulation",
    "src/logging_system",
    "src/evaluation",
    "src/storage",
    "src/visualizer",
)


def test_required_directories_exist() -> None:
    missing = [
        path
        for path in REQUIRED_DIRECTORIES
        if not (REPOSITORY_ROOT / path).is_dir()
    ]
    assert not missing, f"Missing required directories: {missing}"


def test_required_files_exist() -> None:
    missing = [path for path in REQUIRED_FILES if not (REPOSITORY_ROOT / path).is_file()]
    assert not missing, f"Missing required files: {missing}"


def test_source_directories_are_packages() -> None:
    missing = [
        path
        for path in PACKAGE_DIRECTORIES
        if not (REPOSITORY_ROOT / path / "__init__.py").is_file()
    ]
    assert not missing, f"Missing package initializers: {missing}"
