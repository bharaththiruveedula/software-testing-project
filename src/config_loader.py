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

    scenarios = []
    for s_dict in data["scenarios"]:
        name = s_dict.get("name", "Unnamed Scenario")
        url = s_dict.get("url")
        if not url:
            raise ValueError(f"Scenario '{name}' is missing required 'url'")
        
        # We explicitly look up fields and apply defaults if missing
        scenario = TestScenario(
            name=name,
            url=url,
            model=s_dict.get("model", "default"),
            users=s_dict.get("users", 10),
            prompt=s_dict.get("prompt", "Explain software testing like I'm five."),
            schema=s_dict.get("schema"),
            timeout=float(s_dict.get("timeout", 60.0)),
            temperature=float(s_dict.get("temperature", 0.7)),
            stagger=float(s_dict.get("stagger", 0.0)),
            threshold=float(s_dict.get("threshold", 0.95)),
        )
        scenarios.append(scenario)
        
    return scenarios
