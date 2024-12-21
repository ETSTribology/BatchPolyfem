import os
import json
from dotenv import load_dotenv
from jsonschema import validate, ValidationError, SchemaError
import logging
from rich.logging import RichHandler

class Config:
    _instance = None

    def __new__(cls, config_file='config.json', schema_file='config.schema.json', verbose=False):
        if not cls._instance:
            print("Creating new Config instance")
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.settings = {}
            cls._instance.logger = cls._instance.setup_logging(verbose)
            cls._instance.load_settings(config_file, schema_file)
        else:
            print("Using existing Config instance")
        return cls._instance

    def setup_logging(self, verbose: bool):
        """
        Sets up logging configuration with RichHandler.
        If verbose is True, set log level to DEBUG, else INFO.
        """
        log_level = logging.DEBUG if verbose else logging.INFO

        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        logger = logging.getLogger("rich")
        return logger

    def load_settings(self, config_file: str, schema_file: str):
        # Load environment variables from .env file
        load_dotenv()

        # Load settings from environment variables
        try:
            self.settings = {
                # General Settings
                'database_url': os.getenv('DATABASE_URL', 'localhost:5432'),
                'api_key': os.getenv('API_KEY', '1234567890abcdef'),
                'debug': os.getenv('DEBUG', 'True').lower() in ['true', '1', 't'],

                # MinIO Environment Variables
                'minio': {
                    'MINIO_ROOT_USER': os.getenv('MINIO_ROOT_USER', 'minioadmin'),
                    'MINIO_ROOT_PASSWORD': os.getenv('MINIO_ROOT_PASSWORD', 'minioadmin'),
                    'MINIO_PROMETHEUS_AUTH_TYPE': os.getenv('MINIO_PROMETHEUS_AUTH_TYPE', 'public'),
                },

                # Zookeeper Environment Variables
                'zookeeper': {
                    'ZOOKEEPER_CLIENT_PORT': int(os.getenv('ZOOKEEPER_CLIENT_PORT', '2181')),
                    'ZOOKEEPER_TICK_TIME': int(os.getenv('ZOOKEEPER_TICK_TIME', '2000')),
                },

                # Kafka Environment Variables
                'kafka': {
                    'KAFKA_BROKER_ID': int(os.getenv('KAFKA_BROKER_ID', '1')),
                    'KAFKA_ZOOKEEPER_CONNECT': os.getenv('KAFKA_ZOOKEEPER_CONNECT', 'zookeeper:2181'),
                    'KAFKA_ADVERTISED_LISTENERS': os.getenv('KAFKA_ADVERTISED_LISTENERS', 'PLAINTEXT://localhost:9092'),
                    'KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR': int(os.getenv('KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR', '1')),
                },

                # MongoDB Environment Variables
                'mongo': {
                    'MONGO_INITDB_ROOT_USERNAME': os.getenv('MONGO_INITDB_ROOT_USERNAME', 'root'),
                    'MONGO_INITDB_ROOT_PASSWORD': os.getenv('MONGO_INITDB_ROOT_PASSWORD', 'example'),
                },

                # Qdrant Environment Variables
                'qdrant': {
                    'QDRANT__SERVICE__HTTP_PORT': int(os.getenv('QDRANT__SERVICE__HTTP_PORT', '6333')),
                    'QDRANT__COLLECTIONS__DEFAULT__NAME': os.getenv('QDRANT__COLLECTIONS__DEFAULT__NAME', 'default'),
                },

                # Grafana Environment Variables
                'grafana': {
                    'GF_SECURITY_ADMIN_USER': os.getenv('GF_SECURITY_ADMIN_USER', 'admin'),
                    'GF_SECURITY_ADMIN_PASSWORD': os.getenv('GF_SECURITY_ADMIN_PASSWORD', 'admin'),
                },
            }
            self.logger.info("Successfully loaded settings from environment variables.")
        except ValueError as ve:
            self.logger.error(f"Type conversion error: {ve}")
            raise

        # Load and validate against JSON Schema
        try:
            with open(schema_file, "r") as f:
                schema = json.load(f)
            self.logger.debug(f"Successfully loaded JSON schema from {schema_file}")
        except FileNotFoundError:
            self.logger.error(f"Schema file not found: {schema_file}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format in schema file {schema_file}: {e}")
            raise

        try:
            validate(instance=self.settings, schema=schema)
            self.logger.info("Configuration validation successful.")
        except ValidationError as ve:
            self.logger.error(f"Configuration validation error: {ve.message}")
            raise
        except SchemaError as se:
            self.logger.error(f"Invalid JSON Schema: {se.message}")
            raise

    def get_setting(self, key: str):
        return self.settings.get(key)

    def set_setting(self, key: str, value):
        self.settings[key] = value
