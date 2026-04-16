"""
Configuration Loader Module

Loads multi-scenario load test configurations from YAML files.
"""

import yaml
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestScenario:
    name: str
    url: str
    model: str
    users: int
    prompt: str = "Explain software testing like I'm five."
    schema: Optional[str] = None
    timeout: float = 60.0
    temperature: float = 0.7
    stagger: float = 0.0
    threshold: float = 0.95


def load_config(file_path: str) -> List[TestScenario]:
    """
    Parse a YAML configuration file into a list of TestScenario objects.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to read YAML config {file_path}: {e}")

    if not data or "scenarios" not in data:
        raise ValueError("Invalid config format: missing 'scenarios' key")

    raw_scenarios = data["scenarios"]
    if not isinstance(raw_scenarios, list):
        raise ValueError("Invalid config format: 'scenarios' must be a list")

    scenarios = []
    for s_dict in raw_scenarios:
        if not isinstance(s_dict, dict):
            raise ValueError("Each scenario must be a mapping/object")

        name = s_dict.get("name", "Unnamed Scenario")
        url = s_dict.get("url")
        if not url:
            raise ValueError(f"Scenario '{name}' is missing required 'url'")

        users = int(s_dict.get("users", 10))
        timeout = float(s_dict.get("timeout", 60.0))
        temperature = float(s_dict.get("temperature", 0.7))
        stagger = float(s_dict.get("stagger", 0.0))
        threshold = float(s_dict.get("threshold", 0.95))

        if users <= 0:
            raise ValueError(f"Scenario '{name}' has invalid 'users' value: must be > 0")
        if timeout <= 0:
            raise ValueError(f"Scenario '{name}' has invalid 'timeout' value: must be > 0")
        if stagger < 0:
            raise ValueError(f"Scenario '{name}' has invalid 'stagger' value: must be >= 0")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Scenario '{name}' has invalid 'threshold' value: must be between 0 and 1")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"Scenario '{name}' has invalid 'temperature' value: must be between 0 and 2")
        
        # We explicitly look up fields and apply defaults if missing
        scenario = TestScenario(
            name=name,
            url=url,
            model=s_dict.get("model", "default"),
            users=users,
            prompt=s_dict.get("prompt", "Explain software testing like I'm five."),
            schema=s_dict.get("schema"),
            timeout=timeout,
            temperature=temperature,
            stagger=stagger,
            threshold=threshold,
        )
        scenarios.append(scenario)
        
    return scenarios
