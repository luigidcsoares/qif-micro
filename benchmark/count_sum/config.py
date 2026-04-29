"""Configuration management for count-sum benchmarks

Defines experiment configurations and provides utilities for loading
and saving configurations from/to YAML files.
"""
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """Configuration for a count-sum benchmark experiment.

    Parameters
    ----------
    n_entries : int, optional
        Number of entries in the dataset (default: 100).

    n_cat : int, optional
        Domain size for the ``cat`` attribute (default: 10).

    n_num : int, optional
        Domain size for the ``num`` attribute (default: 10).

    iterations : int, optional
        Number of iterations to repeat (default: 3).
    """
    n_entries: int = 100
    n_cat: int = 10
    n_num: int = 10
    iterations: int = 3
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        return ExperimentConfig(**data)


def load_scenarios_from_yaml(
    filepath: str | Path,
) -> list[tuple[str, ExperimentConfig]]:
    """
    Load named scenario configurations from a YAML file.

    Parameters
    ----------
    filepath : str | Path
        Path to YAML file containing scenario definitions.

    Returns
    -------
    list[tuple[str, ExperimentConfig]]
        List of (scenario_name, config) tuples.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.

    yaml.YAMLError
        If the file is not valid YAML.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Scenario file not found: {filepath}")

    with open(filepath) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("YAML must be a list of scenario objects")

    scenarios = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each scenario must be a dict")

        if "name" not in item:
            raise ValueError("Each scenario must have a 'name'")

        if "config" not in item:
            raise ValueError("Each scenario must have a 'config'")

        name = item["name"]
        cfg = ExperimentConfig.from_dict(item["config"])
        scenarios.append((name, cfg))

    return scenarios


def discover_yaml_files(directory: str | Path) -> list[Path]:
    """
    Discover all YAML files in a directory.

    Parameters
    ----------
    directory : str | Path
        Path to directory to search.

    Returns
    -------
    list[Path]
        Sorted list of YAML file paths in the directory.

    Raises
    ------
    ValueError
        If the path is not a directory or no YAML files found.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    yaml_files = sorted(directory.glob("*.yaml"))
    if not yaml_files:
        raise ValueError(f"No YAML files found in {directory}")

    return yaml_files


def load_multiple_scenarios(
    sources: list[str | Path],
) -> list[tuple[str, ExperimentConfig]]:
    """
    Load scenarios from multiple YAML files or directories.

    Parameters
    ----------
    sources : list[str | Path]
        List of paths, each being either:
        - A YAML file path
        - A directory path (auto-discovers all YAML files)

    Returns
    -------
    list[tuple[str, ExperimentConfig]]
        List of (scenario_name, config) tuples from all sources.

    Raises
    ------
    ValueError
        If any source is invalid or no scenarios found.
    """
    all_scenarios = []

    for source in sources:
        source_path = Path(source)

        if source_path.is_file() and source_path.suffix == ".yaml":
            # Load from single YAML file
            scenarios = load_scenarios_from_yaml(source_path)
            all_scenarios.extend(scenarios)

        elif source_path.is_dir():
            # Discover and load all YAML files from directory
            yaml_files = discover_yaml_files(source_path)
            for yaml_file in yaml_files:
                scenarios = load_scenarios_from_yaml(yaml_file)
                all_scenarios.extend(scenarios)

        else:
            raise ValueError(
                f"Invalid scenario path (not a .yaml file or directory): "
                f"{source}"
            )

    if not all_scenarios:
        raise ValueError("No scenarios loaded from any source")

    return all_scenarios
