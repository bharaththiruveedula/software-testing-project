"""
Unit tests for Configuration Loader Module.
"""

import pytest
from config_loader import load_config, TestScenario as _TestScenario


def test_load_valid_config(tmp_path):
    """Test loading a complete, valid config."""
    yaml_content = """
    scenarios:
      - name: "Test Scenario 1"
        url: "http://test-url"
        model: "gpt-4"
        users: 10
        timeout: 30.5
        schema: "test_schema.json"
        stagger: 1000
    """
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    scenarios = load_config(str(config_file))
    assert len(scenarios) == 1
    
    s = scenarios[0]
    assert isinstance(s, _TestScenario)
    assert s.name == "Test Scenario 1"
    assert s.url == "http://test-url"
    assert s.model == "gpt-4"
    assert s.users == 10
    assert s.timeout == 30.5
    assert s.schema == "test_schema.json"
    assert s.stagger == 1000.0


def test_load_config_defaults(tmp_path):
    """Test that missing optional fields get sensible defaults."""
    yaml_content = """
    scenarios:
      - url: "http://test-url"
    """
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    scenarios = load_config(str(config_file))
    assert len(scenarios) == 1
    
    s = scenarios[0]
    assert s.name == "Unnamed Scenario"
    assert s.url == "http://test-url"
    assert s.model == "default"
    assert s.users == 10
    assert s.timeout == 60.0
    assert s.temperature == 0.7
    assert s.schema is None


def test_missing_url_raises_error(tmp_path):
    """Test that omitting URL raises an error."""
    yaml_content = """
    scenarios:
      - name: "Broken"
        users: 5
    """
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    with pytest.raises(ValueError, match="is missing required 'url'"):
        load_config(str(config_file))


def test_missing_scenarios_key(tmp_path):
    """Test that missing root scenarios key raises error."""
    yaml_content = """
    other: "value"
    """
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    with pytest.raises(ValueError, match="missing 'scenarios' key"):
        load_config(str(config_file))


def test_invalid_yaml(tmp_path):
    """Test behavior with malformed YAML."""
    # Use invalid YAML syntax (unclosed quote)
    yaml_content = 'scenarios:\n  - name: "broken\n    url: http'
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    with pytest.raises(ValueError, match="Failed to read YAML config"):
        load_config(str(config_file))


def test_invalid_users_raises_error(tmp_path):
    yaml_content = """
    scenarios:
      - name: "Bad users"
        url: "http://test-url"
        users: 0
    """
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    with pytest.raises(ValueError, match="invalid 'users' value"):
        load_config(str(config_file))


def test_invalid_threshold_raises_error(tmp_path):
    yaml_content = """
    scenarios:
      - name: "Bad threshold"
        url: "http://test-url"
        threshold: 1.2
    """
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    with pytest.raises(ValueError, match="invalid 'threshold' value"):
        load_config(str(config_file))


def test_invalid_stagger_raises_error(tmp_path):
    yaml_content = """
    scenarios:
      - name: "Bad stagger"
        url: "http://test-url"
        stagger: -1
    """
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    with pytest.raises(ValueError, match="invalid 'stagger' value"):
        load_config(str(config_file))


def test_scenarios_must_be_list(tmp_path):
    yaml_content = """
    scenarios:
      name: "not-a-list"
    """
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    with pytest.raises(ValueError, match="'scenarios' must be a list"):
        load_config(str(config_file))
