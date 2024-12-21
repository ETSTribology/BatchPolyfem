import pytest
import json
import logging
from unittest.mock import Mock, patch
from polyfem_runner import PolyfemRunner

# Mock configuration for PolyfemRunner
MOCK_POLYFEM_CONFIG = {
    "polyfem_path": "/path/to/polyfem",
    "mode": "local",
    "logger": logging.getLogger("PolyfemRunnerTest")
}

MOCK_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "parameter1": {"type": "string"},
        "parameter2": {"type": "integer"}
    },
    "required": ["parameter1", "parameter2"]
}

def test_parse_and_validate_json():
    runner = PolyfemRunner(**MOCK_POLYFEM_CONFIG)

    # Mock valid JSON file
    valid_json = {"parameter1": "value1", "parameter2": 10}
    with patch("builtins.open", patch("json.load", return_value=valid_json)):
        result = runner.parse_and_validate_json("config.json", MOCK_JSON_SCHEMA)
        assert result == valid_json, "JSON parsing and validation failed for valid input."

    # Mock invalid JSON file
    invalid_json = {"parameter1": "value1"}  # Missing "parameter2"
    with patch("builtins.open", patch("json.load", return_value=invalid_json)):
        with pytest.raises(Exception):
            runner.parse_and_validate_json("config.json", MOCK_JSON_SCHEMA)

def test_modify_config():
    runner = PolyfemRunner(**MOCK_POLYFEM_CONFIG)

    original_config = {"parameter1": "value1", "parameter2": 10}
    updates = {"parameter2": 20, "parameter3": "new_value"}

    result = runner.modify_config(original_config, updates)

    assert result["parameter2"] == 20, "Failed to update existing parameter."
    assert "parameter3" not in result, "Added unexpected parameter to configuration."

def test_save_config():
    runner = PolyfemRunner(**MOCK_POLYFEM_CONFIG)
    config_data = {"parameter1": "value1", "parameter2": 10}

    with patch("builtins.open", patch("json.dump")) as mock_dump:
        runner.save_config(config_data, "config.json")
        mock_dump.assert_called_once_with(config_data, mock_dump.return_value, indent=4)

def test_run_simulation():
    runner = PolyfemRunner(**MOCK_POLYFEM_CONFIG)
    valid_config = {"parameter1": "value1", "parameter2": 10}

    with patch("polyfem_runner.PolyfemRunner.parse_and_validate_json", return_value=valid_config), \
         patch("polyfem_runner.PolyfemRunner.modify_config", return_value=valid_config), \
         patch("subprocess.run", return_value=Mock(returncode=0, stdout="Simulation completed", stderr="")) as mock_run:

        runner.run_simulation("config.json", MOCK_JSON_SCHEMA)
        mock_run.assert_called_once()

    # Test simulation failure
    with patch("subprocess.run", return_value=Mock(returncode=1, stdout="", stderr="Error occurred")) as mock_run:
        with pytest.raises(RuntimeError):
            runner.run_simulation("config.json", MOCK_JSON_SCHEMA)
