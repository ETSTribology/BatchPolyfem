import pytest
from pymongo import MongoClient
from mongo_connect import MongoConnect
from kafka_connect import KafkaConnect

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

def test_mongo_connection():
    mongo = MongoConnect(MOCK_CONFIG)
    assert mongo.client is not None, "MongoDB client is not initialized."

def test_list_databases():
    mongo = MongoConnect(MOCK_CONFIG)
    databases = mongo.list_databases()
    assert isinstance(databases, list), "Databases should be returned as a list."

def test_create_and_drop_database():
    mongo = MongoConnect(MOCK_CONFIG)
    test_db_name = "test_db"

    # Create database
    mongo.create_database(test_db_name)
    databases = mongo.list_databases()
    assert test_db_name in databases, "Test database was not created."

    # Drop database
    mongo.drop_database(test_db_name)
    databases = mongo.list_databases()
    assert test_db_name not in databases, "Test database was not dropped."

def test_create_and_drop_collection():
    mongo = MongoConnect(MOCK_CONFIG)
    test_db_name = "test_db"
    test_collection_name = "test_collection"

    # Create collection
    mongo.create_database(test_db_name)
    mongo.create_collection(test_db_name, test_collection_name)
    collections = mongo.list_collections(test_db_name)
    assert test_collection_name in collections, "Test collection was not created."

    # Drop collection
    mongo.drop_collection(test_db_name, test_collection_name)
    collections = mongo.list_collections(test_db_name)
    assert test_collection_name not in collections, "Test collection was not dropped."

    # Clean up
    mongo.drop_database(test_db_name)

def test_insert_and_find_document():
    mongo = MongoConnect(MOCK_CONFIG)
    test_db_name = "test_db"
    test_collection_name = "test_collection"
    test_document = {"key": "value"}

    # Create collection
    mongo.create_database(test_db_name)
    mongo.create_collection(test_db_name, test_collection_name)

    # Insert document
    inserted_id = mongo.insert_document(test_db_name, test_collection_name, test_document)
    assert inserted_id is not None, "Document was not inserted."

    # Find document
    documents = mongo.find_documents(test_db_name, test_collection_name, {"key": "value"})
    assert len(documents) > 0, "Document was not found."

    # Clean up
    mongo.drop_database(test_db_name)

def test_update_and_delete_documents():
    mongo = MongoConnect(MOCK_CONFIG)
    test_db_name = "test_db"
    test_collection_name = "test_collection"
    test_document = {"key": "value"}

    # Create collection
    mongo.create_database(test_db_name)
    mongo.create_collection(test_db_name, test_collection_name)

    # Insert document
    mongo.insert_document(test_db_name, test_collection_name, test_document)

    # Update document
    updated_count = mongo.update_documents(test_db_name, test_collection_name, {"key": "value"}, {"$set": {"key": "new_value"}})
    assert updated_count > 0, "Document was not updated."

    # Delete document
    deleted_count = mongo.delete_documents(test_db_name, test_collection_name, {"key": "new_value"})
    assert deleted_count > 0, "Document was not deleted."

    # Clean up
    mongo.drop_database(test_db_name)

def test_kafka_list_connectors():
    kafka = KafkaConnect(MOCK_KAFKA_CONFIG)
    with pytest.raises(Exception):
        connectors = kafka.list_connectors()
        assert isinstance(connectors, list), "Connectors should be returned as a list."

def test_kafka_create_and_delete_connector():
    kafka = KafkaConnect(MOCK_KAFKA_CONFIG)
    test_connector_name = "test_connector"
    test_connector_config = {
        "connector.class": "FileStreamSinkConnector",
        "tasks.max": "1",
        "file": "/tmp/test.txt",
        "topics": "test_topic"
    }

    # Create connector
    with pytest.raises(Exception):
        result = kafka.create_connector(test_connector_name, test_connector_config)
        assert result["name"] == test_connector_name, "Connector was not created."

    # Delete connector
    with pytest.raises(Exception):
        result = kafka.delete_connector(test_connector_name)
        assert "error_code" not in result, "Connector was not deleted."