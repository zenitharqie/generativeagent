from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = REPOSITORY_ROOT / "main.py"


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MAIN_SCRIPT), *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


def test_main_runs_without_arguments() -> None:
    result = run_cli()

    assert result.returncode == 0
    assert "repository scaffold is ready" in result.stdout
    assert result.stderr == ""


def test_main_help_runs() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "validate-config" in result.stdout
    assert result.stderr == ""


def test_validate_config_is_an_explicit_placeholder() -> None:
    result = run_cli("validate-config")

    assert result.returncode == 0
    assert "not implemented yet" in result.stdout
    assert result.stderr == ""
