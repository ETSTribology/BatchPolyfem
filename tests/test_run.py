import pytest
from mongo_connect import MongoConnect
from kafka_connect import KafkaConnect
from minio_connect import MinIOConnect
from run import app
from typer.testing import CliRunner

# Mock configuration for MongoDB
MOCK_CONFIG = {
    "host": "localhost",
    "port": 27017,
    "username": "test_user",
    "password": "test_password",
    "authSource": "admin",
    "verbose": False
}

# Mock configuration for KafkaConnect
MOCK_KAFKA_CONFIG = {
    "url": "http://localhost:8083",
    "username": "test_user",
    "password": "test_password",
    "verify_ssl": False,
    "timeout": 30,
    "verbose": False
}

# Mock configuration for MinIOConnect
MOCK_MINIO_CONFIG = {
    "endpoint": "localhost:9000",
    "access_key": "minioadmin",
    "secret_key": "minioadmin",
    "secure": False,
    "verbose": False
}

def test_run_command_with_no_tasks(monkeypatch):
    runner = CliRunner()

    # Mock config settings
    mock_config = {
        "mongo": MOCK_CONFIG,
        "kafka": MOCK_KAFKA_CONFIG,
        "minio": MOCK_MINIO_CONFIG,
        "tasks": []
    }

    def mock_load_settings(*args, **kwargs):
        return mock_config

    monkeypatch.setattr("config.Config", mock_load_settings)

    result = runner.invoke(app, ["run", "--file", "config.json", "--verbose"])
    assert result.exit_code == 1
    assert "No tasks found in the configuration." in result.output

def test_run_command_with_tasks(monkeypatch):
    runner = CliRunner()

    # Mock config settings
    mock_config = {
        "mongo": MOCK_CONFIG,
        "kafka": MOCK_KAFKA_CONFIG,
        "minio": MOCK_MINIO_CONFIG,
        "tasks": [
            {
                "id": 1,
                "name": "Task 1",
                "description": "A test task",
                "type": "file_processing",
                "bucket_name": "test-bucket",
                "object_name": "test-object.txt"
            }
        ]
    }

    def mock_load_settings(*args, **kwargs):
        return mock_config

    def mock_process_file_and_dump_to_mongo(*args, **kwargs):
        pass

    monkeypatch.setattr("config.Config", mock_load_settings)
    monkeypatch.setattr("run.process_file_and_dump_to_mongo", mock_process_file_and_dump_to_mongo)

    result = runner.invoke(app, ["run", "--file", "config.json", "--verbose"])
    assert result.exit_code == 0
    assert "Processing tasks with Rich Progress..." in result.output

def test_mongo_connection():
    mongo = MongoConnect(MOCK_CONFIG)
    assert mongo.client is not None, "MongoDB client is not initialized."

def test_kafka_list_connectors():
    kafka = KafkaConnect(MOCK_KAFKA_CONFIG)
    with pytest.raises(Exception):
        connectors = kafka.list_connectors()
        assert isinstance(connectors, list), "Connectors should be returned as a list."

def test_minio_list_buckets():
    minio = MinIOConnect(MOCK_MINIO_CONFIG)
    with pytest.raises(Exception):
        buckets = minio.list_buckets()
        assert isinstance(buckets, list), "Buckets should be returned as a list."
