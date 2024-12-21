import os
import json
import pytest
from unittest.mock import patch, mock_open
from jsonschema.exceptions import ValidationError
from config import Config

def test_singleton_behavior():
    config1 = Config(config_file='config.json', schema_file='config.schema.json', verbose=False)
    config2 = Config(config_file='config.json', schema_file='config.schema.json', verbose=False)
    assert config1 is config2, "Singleton behavior failed."

def test_load_settings_from_env():
    with patch.dict(os.environ, {
        'DATABASE_URL': 'test_db_url',
        'API_KEY': 'test_api_key',
        'DEBUG': 'True',
        'MINIO_ROOT_USER': 'test_minio_user',
        'MINIO_ROOT_PASSWORD': 'test_minio_password'
    }):
        config = Config(config_file='config.json', schema_file='config.schema.json', verbose=False)
        assert config.get_setting('database_url') == 'test_db_url'
        assert config.get_setting('api_key') == 'test_api_key'
        assert config.get_setting('debug') is True
        assert config.get_setting('minio')['MINIO_ROOT_USER'] == 'test_minio_user'

def test_schema_validation():
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "database_url": {"type": "string"},
            "api_key": {"type": "string"},
            "debug": {"type": "boolean"}
        },
        "required": ["database_url", "api_key", "debug"]
    }

    valid_config = {
        "database_url": "test_db_url",
        "api_key": "test_api_key",
        "debug": True
    }

    invalid_config = {
        "api_key": "test_api_key",
        "debug": True
    }

    with patch("builtins.open", mock_open(read_data=json.dumps(schema))):
        # Test valid config
        with patch("config.json.load", return_value=valid_config):
            config = Config(config_file='config.json', schema_file='config.schema.json', verbose=False)
            assert config.get_setting('database_url') == "test_db_url"

        # Test invalid config
        with patch("config.json.load", return_value=invalid_config):
            with pytest.raises(ValidationError):
                Config(config_file='config.json', schema_file='config.schema.json', verbose=False)

def test_get_and_set_setting():
    config = Config(config_file='config.json', schema_file='config.schema.json', verbose=False)
    config.set_setting('new_key', 'new_value')
    assert config.get_setting('new_key') == 'new_value'
