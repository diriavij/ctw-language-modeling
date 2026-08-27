"""Shared locations for generated experiment artifacts."""

from pathlib import Path


EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
FIGURES_DIR = EXPERIMENTS_DIR / "figures"


def ensure_artifact_dirs() -> None:
    """Create artifact directories before an experiment writes output."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def result_path(filename: str) -> str:
    ensure_artifact_dirs()
    return str(RESULTS_DIR / filename)


def figure_path(filename: str) -> str:
    ensure_artifact_dirs()
    return str(FIGURES_DIR / filename)


def figure_for_result(path: str, suffix: str = ".png") -> str:
    """Place a default result's figure in figures/, preserving custom paths."""
    result = Path(path)
    if result.parent.resolve() == RESULTS_DIR.resolve():
        return figure_path(result.stem + suffix)
    return str(result.with_suffix(suffix))
